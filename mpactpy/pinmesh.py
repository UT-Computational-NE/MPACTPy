from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from math import isclose, hypot
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import tempfile
import os
import uuid
import shutil

import openmc
import numpy as np

from mpactpy.material import Material
from mpactpy.utils import relative_round, allclose, list_to_str, ROUNDING_RELATIVE_TOLERANCE as TOL, \
                          temporary_environment, equal_thickness_regions, RadialDivisionType, subdivide_ring


# =======================================
# Helper functions for overlay processing
# =======================================

def _process_centroid_batch(args: Tuple) -> List[Material]:
    """ Processes a batch of centroid points to determine material assignments in parallel.

    Parameters
    ----------
    args : Tuple
        A tuple containing:
            points_batch : list of tuple
                A list of points (e.g., coordinates) to be processed.
            geometry : openmc.Geometry
                The OpenMC geometry to query for material assignments.
            mat_specs : any
                Material specifications or mapping information needed for material assignment.

    Returns
    -------
    List[Material]
        A list of Material objects corresponding to each centroid in the batch.
    """
    points_batch, geometry, mat_specs = args

    results = []
    for point in points_batch:
        mat = Material.from_openmc_geometry_point(point, geometry, mat_specs)
        results.append(mat)
    return results


def _process_homogenized_batch(args: Tuple) -> List[Material]:
    """ Processes a batch of elements to determine material assignments in parallel.

    Parameters
    ----------
    args : Tuple
        A tuple containing:
            elements_batch : List[tuple[int | None, float]]
                A list of elements to be processed for homogenized material assignment.
            geometry : openmc.Geometry
                The OpenMC geometry to query for material assignments.
            mat_specs : any
                Material specifications or mapping information needed for material assignment.
            mix_policy : any
                Mixing policy or additional information for homogenization.

    Returns
    -------
    List[Material]
        A list of Material objects corresponding to each element in the batch.
    """
    elements_batch, geometry, mat_specs, mix_policy = args

    results = []
    for element in elements_batch:
        mat = Material.from_openmc_geometry_element(element, geometry, mat_specs, mix_policy)
        results.append(mat)
    return results


def _materials_at_centroids(centroids: np.ndarray,
                            geometry: openmc.Geometry,
                            overlay_policy: PinMesh.OverlayPolicy) -> List[Material]:
    """ Determines material assignments at specified centroids

    Parameters
    ----------
    centroids : numpy.ndarray
        Array of centroid coordinates to process.
    geometry : openmc.Geometry
        The OpenMC geometry used to determine material assignments.
    overlay_policy : PinMesh.OverlayPolicy
        Policy object specifying overlay options.

    Returns
    -------
    List[Material]
        A list of Material objects corresponding to each centroid.
    """

    # Run overlay in serial
    if overlay_policy.num_procs <= 1:
        materials = []
        for point in centroids:
            mat = Material.from_openmc_geometry_point(point, geometry, overlay_policy.mat_specs)
            materials.append(mat)
        return materials

    # Run overlay in parallel
    chunks    = np.array_split(centroids, overlay_policy.num_procs)
    args_list = [(chunk, geometry, overlay_policy.mat_specs) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=overlay_policy.num_procs) as executor:
        batch_results = list(executor.map(_process_centroid_batch, args_list))

    materials = [item for sublist in batch_results for item in sublist]
    return materials


def _materials_in_elements(elements: List[tuple[int | None, float]],
                           geometry: openmc.Geometry,
                           overlay_policy: PinMesh.OverlayPolicy) -> List[Material]:
    """ Determines material assignments in specified homogenized elements

    Parameters
    ----------
    elements : List[tuple[int | None, float]]
        List of material volume elements to process.
    geometry : openmc.Geometry
        The OpenMC geometry used to determine material assignments.
    overlay_policy : PinMesh.OverlayPolicy
        Policy object specifying overlay options.

    Returns
    -------
    List[Material]
        A list of Material objects corresponding to each element.
    """

    # Run overlay in serial
    if overlay_policy.num_procs <= 1:
        materials = []
        for element in elements:
            mat = Material.from_openmc_geometry_element(element, geometry, overlay_policy.mat_specs, overlay_policy.mix_policy)
            materials.append(mat)
        return materials

    # Run overlay in parallel
    chunk_indices = np.array_split(range(len(elements)), overlay_policy.num_procs)
    chunks        = [[elements[i] for i in indices] for indices in chunk_indices if len(indices) > 0]
    args_list     = [(chunk, geometry, overlay_policy.mat_specs, overlay_policy.mix_policy) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=overlay_policy.num_procs) as executor:
        batch_results = list(executor.map(_process_homogenized_batch, args_list))

    materials = []
    for batch_result in batch_results:
        materials.extend(batch_result)

    return materials

# =============================================================================
# Class definitions
# =============================================================================
class PinMesh(ABC):
    """  An abstract class for MPACT model pinmeshes

    A subtle nuance of MPACT that must be addressed regarding pinmeshes is the
    handling of material / cross-section regions (XSRs) that don't lie at least
    partially within the boundaries of a given pinmesh.  Currently, MPACT requires
    the user to NOT specify XSRs (and their materials in the Pin Cards) which don't lie at least
    partially within the bounds of the mesh.  However, having the client keep track
    of which XSRs are and are not partially within the bounds can be very tedious for
    the client.  Therefore, this API will provide helper methods to make this current nuance
    of MPACT transparent to the client.  Meaning, the client need not to concern themselves
    whether their XSR definitions and materials exist within the mesh bounds or not.

    Attributes
    ----------
    number_of_material_regions : int
        The total number of pinmesh material regions (i.e. cross-section regions) regardless
        of whether they fall within the bounds of the pinmesh or not
    zvals : List[float]
        The z-coordinates marking material interfaces within the pin
    ndivz : List[int]
        The number of equally spaced flat source regions to use when
        dividing the grid defined by zval
    pitch : Dict[str, float]
        The pitch of the pin mesh in each axis direction (keys: 'X', 'Y', 'Z')
    regions_inside_bounds = List[int]
        The list of material regions that lie within the bounds of the pinmesh
    """
    _number_of_material_regions: int
    _zvals: List[float]
    _ndivz: List[int]
    _pitch: Dict[str, float]
    _regions_inside_bounds: List[int]
    _cached_hash: Optional[int]

    class Subdivisions(ABC):
        """Base class for pinmesh-specific material-region subdivision specifications.
        """

    @property
    def number_of_material_regions(self) -> int:
        return self._number_of_material_regions

    @property
    def zvals(self) -> List[float]:
        return self._zvals

    @property
    def ndivz(self) -> List[int]:
        return self._ndivz

    @property
    def pitch(self) -> Dict[str, float]:
        return self._pitch

    @property
    def regions_inside_bounds(self) -> List[int]:
        return self._regions_inside_bounds

    @abstractmethod
    def write_to_string(self, prefix: str = "", mpact_ids: Dict[PinMesh, int] = None) -> str:
        """ Method for writing pin mesh to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string
        mpact_ids : Dict[PinMesh, int]
            A collection of PinMeshes and their corresponding MPACT IDs

        Returns
        -------
        str
            The string that represents the pin mesh
        """

    @abstractmethod
    def subdivide(self, subdivisions: PinMesh.Subdivisions) -> Tuple[PinMesh, List[int]]:
        """Return a subdivided mesh and the mapping from new regions to old regions.

        Parameters
        ----------
        subdivisions : PinMesh.Subdivisions
            Pinmesh-specific material-region subdivision specifications.

        Returns
        -------
        Tuple[PinMesh, List[int]]
            The subdivided pin mesh and a list mapping each new material-region
            index to the source material-region index in the original mesh.
        """

    @abstractmethod
    def divide_into_quadrants(self) -> List[List[Tuple[PinMesh, List[int]]]]:
        """Return quadrant pin meshes and their material-region mappings.

        Returns
        -------
        List[List[Tuple[PinMesh, List[int]]]]
            Quadrant meshes and material maps ordered as
            ``[[NW, NE], [SW, SE]]``. Each material map maps a quadrant
            material-region index to the source material-region index in the
            original mesh.
        """

    def set_axial_mesh(self, zvals: List[float] = None, ndivz: List[int] = None) -> None:
        """ A method for setting the axial meshing of a pinmesh

        Parameters
        ----------
        zvals : List[float]
            The z-coordinates marking material interfaces within the pin
        ndivz : List[int]
            The number of equally spaced flat source regions to use when
            dividing the grid defined by zval
        """
        zvals = zvals if zvals else []
        ndivz = ndivz if ndivz else []

        assert len(zvals) > 0, f"len(zvals) = {len(zvals)}"
        assert all(val > 0. for val in zvals), f"zvals = {zvals}"
        assert all(zvals[i-1] < zvals[i] for i in range(1,len(zvals))), f"zvals = {zvals}"
        assert (len(ndivz) == len(zvals)), f"len(ndivz) = {len(ndivz)}, len(zvals) = {len(zvals)}"
        assert all(val > 0 for val in ndivz), f"ndivz = {ndivz}"

        self._zvals = zvals
        self._ndivz = ndivz

        self._set_pitch()
        self._set_number_of_material_regions()
        self._set_regions_inside_bounds()
        self._cached_hash = None


    @abstractmethod
    def _set_pitch(self) -> None:
        """ Method for setting the pinmesh pitches
        """

    @abstractmethod
    def _set_number_of_material_regions(self) -> None:
        """ Method for setting the pinmesh number of material regions
        """

    @abstractmethod
    def _set_regions_inside_bounds(self) -> None:
        """ Method for setting the material regions that are inside the pinmesh bounds
        """

    @dataclass
    class OverlayPolicy():
        """ Data class for specifying how to pinmesh overlays

        Attributes
        ----------
        method : str
            Material mapping method:
            - 'centroid'   : Assigns each region the material located at its centroid.
            - 'homogenized': Computes a volume-weighted mixture of materials within each region.
            Defaults to 'centroid'.
        n_samples : int
            Number of samples (rays) used for estimating material fractions when using
            the 'homogenized' method. Ignored if method is 'centroid'.
        mat_specs : Dict[openmc.Material, Material.MPACTSpecs]
            A mapping from OpenMC materials to corresponding MPACT material specifications.
            If a material is not found in this mapping, `Material.MPACTSpecs()` is used as the default.
        mix_policy : Optional[Material.MixPolicy]
            Policy for how to mix materials. Used only when `method='homogenized'`.
            If a mix_policy is not found, then Material.MixPolicy() is used by default.
        num_procs : int
            Number of processors to use for executing the overlay operation
        """

        method:     str = "centroid"
        n_samples:  int = 10000
        mat_specs:  Dict[openmc.Material, Material.MPACTSpecs] = field(default_factory=dict)
        mix_policy: Optional[Material.MixPolicy] = None
        num_procs:  int = 1

        def __post_init__(self):
            assert self.method in ["centroid", "homogenized"], f"Unknown method: {self.method}"
            assert self.n_samples > 0, f"n_samples = {self.n_samples}"
            assert self.num_procs > 0, f"num_procs = {self.num_procs}"
            self.mix_policy = Material.MixPolicy() if self.mix_policy is None else self.mix_policy

        def allocate_processes(self, num_children: int) -> PinMesh.OverlayPolicy:
            """Allocate process budget among child operations

            Parameters
            ----------
            num_children : int
                Number of child operations to distribute processes among

            Returns
            -------
            PinMesh.OverlayPolicy
                A new policy with processes allocated for child operations
            """
            if self.num_procs <= 1 or num_children <= 1:
                child_policy = deepcopy(self)
                child_policy.num_procs = 1
                return child_policy

            processes_per_child = max(1, self.num_procs // num_children)
            child_policy = deepcopy(self)
            child_policy.num_procs = processes_per_child
            return child_policy


    @abstractmethod
    def overlay(self,
                geometry:       openmc.Geometry,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                overlay_policy: OverlayPolicy = OverlayPolicy(),
    ) -> List[Optional[Material]]:
        """ A method for overlaying an OpenMC geometry over top a MPACTPy PinMesh

        Parameters
        ----------
        geometry : openmc.Geometry
            The OpenMC Geometry to be mapped onto the MPACTPy PinMesh
        offset : Tuple[float, float, float]
            Offset of the OpenMC geometry's lower-left corner relative to the
            MPACT PinMesh lower-left. Default is (0.0, 0.0, 0.0)
        overlay_policy : OverlayPolicy
            A configuration object specifying how a mesh overlay should be done.

        Returns
        -------
        List[Optional[Material]]
            List of materials assigned to each cell in the PinMesh based on
            the overlaid OpenMC geometry.
        """


class RectangularPinMesh(PinMesh):
    """  An MPACT model pin mesh made up of a 3-D rectilinear grid

    Parameters
    ----------
    xvals : List[float]
        The x-coordinates marking material interfaces within the pin
    yvals : List[float]
        The y-coordinates marking material interfaces within the pin
    zvals : List[float]
        The z-coordinates marking material interfaces within the pin
    ndivx : List[int]
        The number of equally spaced flat source regions to use when
        dividing the grid defined by xval
    ndivy : List[int]
        The number of equally spaced flat source regions to use when
        dividing the grid defined by yval
    ndivz : List[int]
        The number of equally spaced flat source regions to use when
        dividing the grid defined by zval

    Attributes
    ----------
    xvals : List[float]
        The x-coordinates marking material interfaces within the pin
    yvals : List[float]
        The y-coordinates marking material interfaces within the pin
    ndivx : List[int]
        The number of equally spaced flat source regions to use when
        dividing the grid defined by xval
    ndivy : List[int]
        The number of equally spaced flat source regions to use when
        dividing the grid defined by yval
    """

    @dataclass
    class Subdivisions(PinMesh.Subdivisions):
        """Material-region subdivision specifications for a rectangular pin mesh.

        Attributes
        ----------
        subd_x : Optional[List[int]]
            X-direction material-region subdivision counts. The list must have
            length ``len(xvals)``. If ``None``, no X-direction material
            subdivision is applied.
        subd_y : Optional[List[int]]
            Y-direction material-region subdivision counts. The list must have
            length ``len(yvals)``. If ``None``, no Y-direction material
            subdivision is applied.
        subd_z : Optional[List[int]]
            Z-direction material-region subdivision counts. The list must have
            length ``len(zvals)``. If ``None``, no Z-direction material
            subdivision is applied.
        """
        subd_x: Optional[List[int]] = None
        subd_y: Optional[List[int]] = None
        subd_z: Optional[List[int]] = None

    @property
    def xvals(self) -> List[float]:
        return self._xvals

    @property
    def yvals(self) -> List[float]:
        return self._yvals

    @property
    def ndivx(self) -> List[int]:
        return self._ndivx

    @property
    def ndivy(self) -> List[int]:
        return self._ndivy


    def __init__(self,
                 xvals:    List[float],
                 yvals:    List[float],
                 zvals:    List[float],
                 ndivx:    List[int],
                 ndivy:    List[int],
                 ndivz:    List[int],
    ):
        assert len(xvals) > 0, f"len(xvals) = {len(xvals)}"
        assert len(yvals) > 0, f"len(yvals) = {len(yvals)}"
        assert all(val > 0. for val in xvals), f"xvals = {xvals}"
        assert all(val > 0. for val in yvals), f"yvals = {yvals}"
        assert all(val > 0. for val in zvals), f"zvals = {zvals}"
        assert all(xvals[i-1] < xvals[i] for i in range(1,len(xvals))), f"xvals = {xvals}"
        assert all(yvals[i-1] < yvals[i] for i in range(1,len(yvals))), f"yvals = {yvals}"
        assert (len(ndivx) == len(xvals)), f"len(ndivx) = {len(ndivx)}, len(xvals) = {len(xvals)}"
        assert (len(ndivy) == len(yvals)), f"len(ndivy) = {len(ndivy)}, len(yvals) = {len(yvals)}"
        assert all(val > 0 for val in ndivx), f"ndivx = {ndivx}"
        assert all(val > 0 for val in ndivy), f"ndivy = {ndivy}"

        self._xvals   = xvals
        self._yvals   = yvals
        self._ndivx   = ndivx
        self._ndivy   = ndivy

        self.set_axial_mesh(zvals, ndivz)


    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, RectangularPinMesh)           and
                allclose(self.xvals, other.xvals, rtol=TOL)     and
                allclose(self.yvals, other.yvals, rtol=TOL)     and
                allclose(self.zvals, other.zvals, rtol=TOL)     and
                allclose(self.ndivx, other.ndivx, rtol=TOL)     and
                allclose(self.ndivy, other.ndivy, rtol=TOL)     and
                allclose(self.ndivz, other.ndivz, rtol=TOL)
               )


    def __hash__(self) -> int:
        if self._cached_hash is None:
            pitches = {key : relative_round(val, TOL) for key, val in self.pitch.items()}
            self._cached_hash = hash((tuple(relative_round(val, TOL) for val in self.xvals),
                                     tuple(relative_round(val, TOL) for val in self.yvals),
                                     tuple(relative_round(val, TOL) for val in self.zvals),
                                     tuple(relative_round(val, TOL) for val in self.ndivx),
                                     tuple(relative_round(val, TOL) for val in self.ndivy),
                                     tuple(relative_round(val, TOL) for val in self.ndivz),
                                     tuple(sorted(pitches)),
                                     self.number_of_material_regions))
        return self._cached_hash

    def write_to_string(self, prefix: str = "", mpact_ids: Dict[PinMesh, int] = None) -> str:

        mpact_id = 1 if mpact_ids is None else mpact_ids[self]
        string =  prefix
        string += f"pinmesh {mpact_id} rec {list_to_str(self._xvals)} / " \
               +  f"{list_to_str(self._yvals)} / {list_to_str(self._zvals)} / " \
               +  f"{list_to_str(self._ndivx)} / {list_to_str(self._ndivy)} / {list_to_str(self._ndivz)}\n"

        return string

    def subdivide(self, subdivisions: PinMesh.Subdivisions) -> Tuple[RectangularPinMesh, List[int]]:
        """Return a copy with additional Cartesian material interfaces.

        Existing FSR subdivision counts are copied to each new material
        subregion. The returned material map uses the standard rectangular pin
        ordering: z-major, then y, with x as the fastest-moving index.

        Parameters
        ----------
        subdivisions : PinMesh.Subdivisions
            Rectangular material-region subdivision specifications.

        Returns
        -------
        Tuple[RectangularPinMesh, List[int]]
            A new pin mesh with the requested material-region subdivisions and
            a list mapping each new material-region index to its original
            material-region index.
        """

        if not isinstance(subdivisions, RectangularPinMesh.Subdivisions):
            raise TypeError(f"Expected RectangularPinMesh.Subdivisions, got {type(subdivisions).__name__}")

        subd_x = [1] * len(self.xvals) if subdivisions.subd_x is None else subdivisions.subd_x
        subd_y = [1] * len(self.yvals) if subdivisions.subd_y is None else subdivisions.subd_y
        subd_z = [1] * len(self.zvals) if subdivisions.subd_z is None else subdivisions.subd_z

        assert len(subd_x) == len(self.xvals), f"len(subd_x) = {len(subd_x)}, expected {len(self.xvals)}"
        assert len(subd_y) == len(self.yvals), f"len(subd_y) = {len(subd_y)}, expected {len(self.yvals)}"
        assert len(subd_z) == len(self.zvals), f"len(subd_z) = {len(subd_z)}, expected {len(self.zvals)}"

        def subdivide_axis(vals: List[float],
                           ndiv: List[int],
                           subd: List[int]) -> Tuple[List[float], List[int], List[int]]:
            new_vals, new_ndiv, material_map = [], [], []
            lower_val = 0.0
            for material_region_index, (upper_val, fsr_divisions, material_divisions) in enumerate(zip(vals, ndiv, subd)):
                for val in equal_thickness_regions(lower_val, upper_val, material_divisions):
                    new_vals.append(val)
                    new_ndiv.append(fsr_divisions)
                    material_map.append(material_region_index)
                lower_val = upper_val
            return new_vals, new_ndiv, material_map

        xvals, ndivx, x_material_map = subdivide_axis(self.xvals, self.ndivx, subd_x)
        yvals, ndivy, y_material_map = subdivide_axis(self.yvals, self.ndivy, subd_y)
        zvals, ndivz, z_material_map = subdivide_axis(self.zvals, self.ndivz, subd_z)

        old_num_x_regions = len(self.xvals)
        old_num_xy_regions = len(self.xvals) * len(self.yvals)
        material_map = [z_index * old_num_xy_regions + y_index * old_num_x_regions + x_index
                        for z_index in z_material_map
                        for y_index in y_material_map
                        for x_index in x_material_map]

        pinmesh = RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz)
        return pinmesh, material_map

    def divide_into_quadrants(self) -> List[List[Tuple[PinMesh, List[int]]]]:
        """Return rectangular quadrant meshes and material-region mappings.

        The split is performed at the midpoint of the X and Y pin widths.
        Returned meshes use local coordinates with their lower-left corner at
        ``(0, 0)``. The quadrant ordering is ``[[NW, NE], [SW, SE]]``.

        Returns
        -------
        List[List[Tuple[PinMesh, List[int]]]]
            Quadrant meshes and material maps ordered as
            ``[[NW, NE], [SW, SE]]``.
        """

        def slice_axis(vals: List[float],
                       ndiv: List[int],
                       lower_bound: float,
                       upper_bound: float) -> Tuple[List[float], List[int], List[int]]:
            new_vals, new_ndiv, material_map = [], [], []
            previous_val = 0.0
            for material_region_index, (val, fsr_divisions) in enumerate(zip(vals, ndiv)):
                overlap_lower = max(previous_val, lower_bound)
                overlap_upper = min(val, upper_bound)
                if overlap_lower < overlap_upper:
                    new_vals.append(overlap_upper - lower_bound)
                    new_ndiv.append(fsr_divisions)
                    material_map.append(material_region_index)
                previous_val = val
            return new_vals, new_ndiv, material_map

        def make_quadrant(x_lower: float,
                          x_upper: float,
                          y_lower: float,
                          y_upper: float) -> Tuple[PinMesh, List[int]]:
            xvals, ndivx, x_material_map = slice_axis(self.xvals, self.ndivx, x_lower, x_upper)
            yvals, ndivy, y_material_map = slice_axis(self.yvals, self.ndivy, y_lower, y_upper)
            zvals, ndivz = self.zvals[:], self.ndivz[:]

            old_num_x_regions = len(self.xvals)
            old_num_xy_regions = len(self.xvals) * len(self.yvals)
            material_map = [z_index * old_num_xy_regions + y_index * old_num_x_regions + x_index
                            for z_index in range(len(self.zvals))
                            for y_index in y_material_map
                            for x_index in x_material_map]

            return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz), material_map

        x_mid = self.xvals[-1] / 2.0
        y_mid = self.yvals[-1] / 2.0

        return [[make_quadrant(0.0,   x_mid,         y_mid, self.yvals[-1]),
                 make_quadrant(x_mid, self.xvals[-1], y_mid, self.yvals[-1])],
                [make_quadrant(0.0,   x_mid,         0.0,   y_mid),
                 make_quadrant(x_mid, self.xvals[-1], 0.0,   y_mid)]]

    def _set_pitch(self) -> None:
        self._pitch = {'X' : self.xvals[-1], 'Y' : self.yvals[-1], 'Z' : self.zvals[-1]}

    def _set_number_of_material_regions(self) -> None:
        self._number_of_material_regions = len(self.xvals)*len(self.yvals)*len(self.zvals)

    def _set_regions_inside_bounds(self) -> None:
        self._regions_inside_bounds = list(range(len(self.xvals)*len(self.yvals)*len(self.zvals)))


    def overlay(self,
                geometry:       openmc.Geometry,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy(),
    ) -> List[Optional[Material]]:

        # Create mesh for overlay operations
        mesh = openmc.RectilinearMesh()
        mesh.x_grid = np.array([0.0] + self.xvals) + offset[0]
        mesh.y_grid = np.array([0.0] + self.yvals) + offset[1]
        mesh.z_grid = np.array([0.0] + self.zvals) + offset[2]

        mesh_shape = (len(self.xvals), len(self.yvals), len(self.zvals))

        # Perform overlay based on the selected method
        if overlay_policy.method == "centroid":
            # Centroid-based overlay using ProcessPoolExecutor
            centroids = mesh.centroids.reshape((-1, 3))
            materials = _materials_at_centroids(centroids, geometry, overlay_policy)
            materials = np.array(materials).reshape(mesh_shape, order='C')
        else:
            # Homogenized overlay using ProcessPoolExecutor
            # Note: For homogenized method, we need a full model for material_volumes
            model = openmc.Model(geometry=geometry)
            model.settings.temperature = {'method': 'interpolation'}
            with temporary_environment("OMP_NUM_THREADS", str(overlay_policy.num_procs)):
                # Create unique temporary workspace for OpenMC Volume calculations
                # This is needed when multiple processes are running this code simultaneously
                unique_id    = str(uuid.uuid4())[:8]
                workspace    = tempfile.mkdtemp(prefix=f"openmc_workspace_{unique_id}_")
                original_cwd = os.getcwd()
                try:
                    os.chdir(workspace)
                    material_volumes = mesh.material_volumes(model, overlay_policy.n_samples)
                finally:
                    os.chdir(original_cwd)
                    shutil.rmtree(workspace, ignore_errors=True)

            elements  = [material_volumes.by_element(i) for i in range(material_volumes.num_elements)]
            materials = _materials_in_elements(elements, geometry, overlay_policy)
            materials = np.array(materials).reshape(mesh_shape, order='F')

        # Convert to MPACT-compatible format
        materials = materials[:, ::-1, :].transpose(2, 1, 0)
        materials = materials.flatten(order='C').tolist()
        return materials



class GeneralCylindricalPinMesh(PinMesh):
    """  An MPACT model pin mesh made up of a concentric cylinders centered at (0,0) with arbitrary pin boundaries

    Parameters
    ----------
    r : List[float]
        Array of radii for indicating the different material interfaces
    xMin : float
        Pin boundary x-min
    xMax : float
        Pin boundary x-max
    yMin : float
        Pin boundary y-min
    yMax : float
        Pin boundary y-max
    ndivr : List[int]
        The number of equal-volume rings to use when dividing the concentric cylinders
        into flat source regions
    ndiva : List[int]
        The number of equal-angle ”pie-slices” to use when dividing each concentric ring
        into flat source regions azimuthally
    zvals : List[float]
        The z-coordinates marking material interfaces within the pin
    ndivz : List[int]
        The number of equally spaced flat source regions to use when
        dividing the grid defined by zval

    Attributes
    ----------
    r : List[float]
        Array of radii for indicating the different material interfaces
    xMin : float
        Pin boundary x-min
    xMax : float
        Pin boundary x-max
    yMin : float
        Pin boundary y-min
    yMax : float
        Pin boundary y-max
    ndivr : List[int]
        The number of equal-volume rings to use when dividing the concentric cylinders
        into flat source regions
    ndiva : List[int]
        The number of equal-angle ”pie-slices” to use when dividing each concentric ring
        into flat source regions azimuthally
    """
    _r_inside_bounds: List[float]
    _ndivr_inside_bounds: List[int]
    _ndiva_inside_bounds: List[int]

    @dataclass
    class Subdivisions(PinMesh.Subdivisions):
        """Material-region subdivision specifications for a general cylindrical pin mesh.

        Attributes
        ----------
        subd_r : Optional[List[int]]
            Radial material-region subdivision counts. The list must have length
            ``len(r) + 1``. If ``None``, no radial material subdivision is applied.
        subd_z : Optional[List[int]]
            Axial material-region subdivision counts. The list must have length
            ``len(zvals)``. If ``None``, no axial material subdivision is applied.
        div_type : Optional[List[RadialDivisionType]]
            Radial subdivision placement rule. ``"equal_thickness"`` spaces
            interfaces uniformly in radius, while ``"equal_volume"`` spaces
            interfaces uniformly in annular area. The list must have length
            ``len(r) + 1``. If ``None``, all radial material regions use
            ``"equal_thickness"``.
        outer_ndivr : int
            Number of radial FSR subdivisions to use for material-region
            subdivisions created from the final implicit outer region. This is
            used only when the outer region is subdivided into additional
            explicit material regions.
        """
        subd_r:   Optional[List[int]] = None
        subd_z:   Optional[List[int]] = None
        div_type: Optional[List[RadialDivisionType]] = None
        outer_ndivr: int = 1

    @property
    def r(self) -> List[float]:
        return self._r

    @property
    def xMin(self) -> float:
        return self._xMin

    @property
    def xMax(self) -> float:
        return self._xMax

    @property
    def yMin(self) -> float:
        return self._yMin

    @property
    def yMax(self) -> float:
        return self._yMax

    @property
    def ndivr(self) -> List[int]:
        return self._ndivr

    @property
    def ndiva(self) -> List[int]:
        return self._ndiva


    def __init__(self,
        r       : List[float],
        xMin    : float,
        xMax    : float,
        yMin    : float,
        yMax    : float,
        zvals   : List[float],
        ndivr   : List[int],
        ndiva   : List[int],
        ndivz   : List[int],
    ):
        assert len(r) > 0, f"len(r) = {len(r)}"
        assert all(val > 0. for val in r), f"r = {r}"
        assert xMin < xMax, f"xMin = {xMin}, xMax = {xMax}"
        assert yMin < yMax, f"yMin = {yMin}, yMax = {yMax}"
        assert len(ndivr) == len(r), f"len(ndivr) = {len(ndivr)}, len(r) = {len(r)}"
        assert len(ndiva) == sum(ndivr)+1, f"len(ndiva) = {len(ndiva)}, sum(ndivr)+1 = {sum(ndivr)+1}"
        assert all(val > 0 for val in ndivr), f"ndivr = {ndivr}"
        assert all(val > 0 for val in ndiva), f"ndiva = {ndiva}"

        self._r       = r
        self._xMin    = xMin
        self._xMax    = xMax
        self._yMin    = yMin
        self._yMax    = yMax
        self._ndivr   = ndivr
        self._ndiva   = ndiva

        self.set_axial_mesh(zvals, ndivz)


    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, GeneralCylindricalPinMesh)   and
                allclose(self.r,     other.r,     rtol=TOL)    and
                isclose( self.xMin,  other.xMin,  rel_tol=TOL) and
                isclose( self.xMax,  other.xMax,  rel_tol=TOL) and
                isclose( self.yMin,  other.yMin,  rel_tol=TOL) and
                isclose( self.yMax,  other.yMax,  rel_tol=TOL) and
                allclose(self.zvals, other.zvals, rtol=TOL)    and
                allclose(self.ndivr, other.ndivr, rtol=TOL)    and
                allclose(self.ndiva, other.ndiva, rtol=TOL)    and
                allclose(self.ndivz, other.ndivz, rtol=TOL)
               )


    def __hash__(self) -> int:
        if self._cached_hash is None:
            pitches = {key : relative_round(val, TOL) for key, val in self.pitch.items()}
            self._cached_hash = hash((relative_round(self.xMin, TOL),
                                      relative_round(self.xMax, TOL),
                                      relative_round(self.yMin, TOL),
                                      relative_round(self.yMax, TOL),
                                      tuple(relative_round(val, TOL) for val in self.r),
                                      tuple(relative_round(val, TOL) for val in self.zvals),
                                      tuple(relative_round(val, TOL) for val in self.ndivr),
                                      tuple(relative_round(val, TOL) for val in self.ndiva),
                                      tuple(relative_round(val, TOL) for val in self.ndivz),
                                      tuple(sorted(pitches)),
                                      self.number_of_material_regions))
        return self._cached_hash


    def write_to_string(self, prefix: str = "", mpact_ids: Dict[PinMesh, int] = None) -> str:

        mpact_id = 1 if mpact_ids is None else mpact_ids[self]
        string = prefix
        string += f"pinmesh {mpact_id} gcyl {list_to_str(self._r_inside_bounds)} / " \
               +  f"{list_to_str([self._xMin, self._xMax, self._yMin, self._yMax])} / " \
               +  f"{list_to_str(self._zvals)} / {list_to_str(self._ndivr_inside_bounds)} / " \
               +  f"{list_to_str(self._ndiva_inside_bounds)} / {list_to_str(self._ndivz)}\n"

        return string

    def subdivide(self, subdivisions: PinMesh.Subdivisions) -> Tuple[GeneralCylindricalPinMesh, List[int]]:
        """Return a copy with additional radial and axial material interfaces.

        Radial subdivision counts apply to the explicit radial material regions
        marked by ``r`` and to the final implicit outer region. The implicit
        outer region is subdivided using the bounding radius of the pin bounds,
        but the bounding radius is not written as a new material interface.

        Axial subdivision counts apply to the material regions marked by
        ``zvals``. Existing FSR subdivision counts are copied to each new
        material subregion.

        Parameters
        ----------
        subdivisions : PinMesh.Subdivisions
            General cylindrical material-region subdivision specifications.

        Returns
        -------
        Tuple[GeneralCylindricalPinMesh, List[int]]
            A new pin mesh with the requested material-region subdivisions and
            a list mapping each new material-region index to its original
            material-region index.
        """

        if not isinstance(subdivisions, GeneralCylindricalPinMesh.Subdivisions):
            raise TypeError(f"Expected GeneralCylindricalPinMesh.Subdivisions, got {type(subdivisions).__name__}")

        n_radial_zones = len(self.r) + 1
        n_axial_zones  = len(self.zvals)

        subd_r   = [1] * n_radial_zones if subdivisions.subd_r is None else subdivisions.subd_r
        subd_z   = [1] * n_axial_zones  if subdivisions.subd_z is None else subdivisions.subd_z
        div_type = ["equal_thickness"] * n_radial_zones if subdivisions.div_type is None else subdivisions.div_type

        assert len(subd_r) == n_radial_zones, f"len(subd_r) = {len(subd_r)}, expected {n_radial_zones}"
        assert len(subd_z) == n_axial_zones, f"len(subd_z) = {len(subd_z)}, expected {n_axial_zones}"
        assert len(div_type) == n_radial_zones, f"len(div_type) = {len(div_type)}, expected {n_radial_zones}"
        assert subdivisions.outer_ndivr > 0, f"outer_ndivr = {subdivisions.outer_ndivr}"

        new_r, new_ndivr, new_ndiva = [], [], []
        radial_material_map = []
        ndiva_index = 0
        inner_radius = 0.0
        for radial_region_index, (outer_radius, ndivr, num_divisions, division_type) in enumerate(zip(
            self.r, self.ndivr, subd_r[:len(self.r)], div_type[:len(self.r)])):

            ndiva_slice = self.ndiva[ndiva_index:ndiva_index + ndivr]
            ndiva_index += ndivr
            for radius in subdivide_ring(inner_radius, outer_radius, num_divisions, division_type):
                new_r.append(radius)
                new_ndivr.append(ndivr)
                new_ndiva.extend(ndiva_slice)
                radial_material_map.append(radial_region_index)
            inner_radius = outer_radius

        outer_ndiva = self.ndiva[ndiva_index]
        outer_region_index = len(self.r)
        bounding_radius = max(hypot(x, y) for x in (self.xMin, self.xMax) for y in (self.yMin, self.yMax))
        if bounding_radius > self.r[-1]:
            for radius in subdivide_ring(self.r[-1], bounding_radius, subd_r[-1], div_type[-1])[:-1]:
                new_r.append(radius)
                new_ndivr.append(subdivisions.outer_ndivr)
                new_ndiva.extend([outer_ndiva] * subdivisions.outer_ndivr)
                radial_material_map.append(outer_region_index)

        new_ndiva.append(outer_ndiva)
        radial_material_map.append(outer_region_index)

        new_zvals, new_ndivz = [], []
        axial_material_map = []
        lower_zval = 0.0
        for axial_region_index, (upper_zval, ndivz, num_divisions) in enumerate(zip(self.zvals, self.ndivz, subd_z)):
            for zval in equal_thickness_regions(lower_zval, upper_zval, num_divisions):
                new_zvals.append(zval)
                new_ndivz.append(ndivz)
                axial_material_map.append(axial_region_index)
            lower_zval = upper_zval

        old_num_radial_regions = len(self.r) + 1
        material_map = [axial_region_index * old_num_radial_regions + radial_region_index
                        for axial_region_index in axial_material_map
                        for radial_region_index in radial_material_map]

        pinmesh = GeneralCylindricalPinMesh(new_r, self.xMin, self.xMax, self.yMin, self.yMax,
                                            new_zvals, new_ndivr, new_ndiva, new_ndivz)
        return pinmesh, material_map

    def divide_into_quadrants(self) -> List[List[Tuple[PinMesh, List[int]]]]:
        """Return general cylindrical quadrant meshes and material-region mappings.

        The split is performed at the midpoint of the X and Y pin bounds.
        Radial and axial material-region definitions are unchanged, so each
        quadrant uses an identity material map. The quadrant ordering is
        ``[[NW, NE], [SW, SE]]``.

        Returns
        -------
        List[List[Tuple[PinMesh, List[int]]]]
            Quadrant meshes and material maps ordered as
            ``[[NW, NE], [SW, SE]]``.
        """

        material_map = list(range(self.number_of_material_regions))

        def make_quadrant(xMin: float,
                          xMax: float,
                          yMin: float,
                          yMax: float) -> Tuple[PinMesh, List[int]]:
            pinmesh = GeneralCylindricalPinMesh(self.r[:], xMin, xMax, yMin, yMax,
                                                self.zvals[:], self.ndivr[:], self.ndiva[:], self.ndivz[:])
            return pinmesh, material_map[:]

        x_mid = (self.xMin + self.xMax) / 2.0
        y_mid = (self.yMin + self.yMax) / 2.0

        return [[make_quadrant(self.xMin, x_mid,     y_mid,    self.yMax),
                 make_quadrant(x_mid,     self.xMax, y_mid,    self.yMax)],
                [make_quadrant(self.xMin, x_mid,     self.yMin, y_mid),
                 make_quadrant(x_mid,     self.xMax, self.yMin, y_mid)]]


    def _set_pitch(self) -> None:
        self._pitch = {'X' : self.xMax - self.xMin,
                       'Y' : self.yMax - self.yMin,
                       'Z' : self.zvals[-1]}

    def _set_number_of_material_regions(self) -> None:
        self._number_of_material_regions = (len(self.r)+1)*len(self.zvals)


    def _set_regions_inside_bounds(self) -> None:

        corners = [hypot(self.xMin, self.yMin), hypot(self.xMin, self.yMax),
                   hypot(self.xMax, self.yMin), hypot(self.xMax, self.yMax)]

        def circle_encloses_box(r):
            return all(corner < r for corner in corners)

        def box_overlaps_circle(r):
            return  not(all(r < corner or isclose(r, corner) for corner in corners) and
                       (self.xMin > 0.0 and self.yMin > 0.0 or
                        self.xMin > 0.0 and self.yMax < 0.0 or
                        self.xMax < 0.0 and self.yMin > 0.0 or
                        self.xMin > 0.0 and self.yMax < 0.0))

        radii_inside_bounds = [i for i,r in enumerate(self.r)
                               if box_overlaps_circle(r) and not circle_encloses_box(r)]

        assert radii_inside_bounds, \
            f"GCYL PinMesh with bounds {self.xMin, self.yMin, self.xMax, self.yMax} " + \
            f"and radii {self.r} has no radial interface which intersects the bounds"

        self._r_inside_bounds       = [self._r[i] for i in radii_inside_bounds]
        self._ndivr_inside_bounds   = [self._ndivr[i] for i in radii_inside_bounds]

        self._ndiva_inside_bounds   = []
        j = 0
        for i, ndivr in enumerate(self.ndivr):
            if i in radii_inside_bounds:
                self._ndiva_inside_bounds.extend(self.ndiva[j:j+ndivr])
            elif i > radii_inside_bounds[-1]:
                break
            j += ndivr

        self._ndiva_inside_bounds.extend(self.ndiva[j:j+1])


        self._regions_inside_bounds = [
            i + z * (len(self.r) + 1)
            for z in range(len(self.zvals))
            for i in radii_inside_bounds + [radii_inside_bounds[-1] + 1]
        ]


    def overlay(self,
                geometry:       openmc.Geometry,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy(),
    ) -> List[Optional[Material]]:

        raise NotImplementedError("Overlay not implemented for GeneralCylindricalPinMesh")
