from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from math import isclose, hypot
from concurrent.futures import ThreadPoolExecutor

import openmc
import numpy as np

from mpactpy.material import Material
from mpactpy.utils import relative_round, allclose, list_to_str, ROUNDING_RELATIVE_TOLERANCE as TOL, \
                          temporary_environment


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

        def get_mpact_specs(self, mat: openmc.Material) -> Material.MPACTSpecs:
            """ Method for retrieving the Material.MPACTSpecs associated with an openmc.Material

            Parameters
            ----------
            material : openmc.Material
                The OpenMC material for which MPACT specifications are being requested.

            Returns
            -------
            Material.MPACTSpecs
                The MPACT specifications associated with the given material, or a default
                `Material.MPACTSpecs()` if no explicit entry is provided.
            """
            return self.mat_specs.get(mat, Material.MPACTSpecs())


    @abstractmethod
    def overlay(self,
                model:          openmc.Model,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                overlay_policy: OverlayPolicy = OverlayPolicy(),
    ) -> List[Optional[Material]]:
        """ A method for overlaying an OpenMC model over top a MPACTPy PinMesh

        Parameters
        ----------
        model : openmc.Model
            The OpenMC Model to be mapped onto the MPACTPy PinMesh
        offset : Tuple[float, float, float]
            Offset of the OpenMC model's lower-left corner relative to the
            MPACT PinMesh lower-left. Default is (0.0, 0.0, 0.0)
        overlay_policy : OverlayPolicy
            A configuration object specifying how a mesh overlay should be done.

        Returns
        -------
        List[Optional[Material]]
            List of materials assigned to each cell in the PinMesh based on
            the overlaid OpenMC model.
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
        pitches = {key : relative_round(val, TOL) for key, val in self.pitch.items()}
        return hash((tuple(relative_round(val, TOL) for val in self.xvals),
                     tuple(relative_round(val, TOL) for val in self.yvals),
                     tuple(relative_round(val, TOL) for val in self.zvals),
                     tuple(relative_round(val, TOL) for val in self.ndivx),
                     tuple(relative_round(val, TOL) for val in self.ndivy),
                     tuple(relative_round(val, TOL) for val in self.ndivz),
                     tuple(sorted(pitches)),
                     self.number_of_material_regions))

    def write_to_string(self, prefix: str = "", mpact_ids: Dict[PinMesh, int] = None) -> str:

        mpact_id = 1 if mpact_ids is None else mpact_ids[self]
        string =  prefix
        string += f"pinmesh {mpact_id} rec {list_to_str(self._xvals)} / " \
               +  f"{list_to_str(self._yvals)} / {list_to_str(self._zvals)} / " \
               +  f"{list_to_str(self._ndivx)} / {list_to_str(self._ndivy)} / {list_to_str(self._ndivz)}\n"

        return string

    def _set_pitch(self) -> None:
        self._pitch = {'X' : self.xvals[-1], 'Y' : self.yvals[-1], 'Z' : self.zvals[-1]}

    def _set_number_of_material_regions(self) -> None:
        self._number_of_material_regions = len(self.xvals)*len(self.yvals)*len(self.zvals)

    def _set_regions_inside_bounds(self) -> None:
        self._regions_inside_bounds = list(range(len(self.xvals)*len(self.yvals)*len(self.zvals)))

    def overlay(self,
                model:          openmc.Model,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy(),
    ) -> List[Optional[Material]]:

        mesh = openmc.RectilinearMesh()
        mesh.x_grid = np.array([0.0] + self.xvals) + offset[0]
        mesh.y_grid = np.array([0.0] + self.yvals) + offset[1]
        mesh.z_grid = np.array([0.0] + self.zvals) + offset[2]

        nx = len(self.xvals)
        ny = len(self.yvals)
        nz = len(self.zvals)

        materials = []
        if overlay_policy.method == "centroid":
            centroids = mesh.centroids.reshape((-1, 3))
            with ThreadPoolExecutor(max_workers=overlay_policy.num_procs) as executor:
                materials = list(executor.map(
                    lambda pt: Material.from_openmc_model_point(pt, model, overlay_policy.mat_specs),
                    centroids))

            # Convert OpenMC's FORTRAN ordering to MPACT expectation
            arr = np.array(materials).reshape((nx, ny, nz), order='F')
            materials = arr.flatten(order='C').tolist()
        else:
            with temporary_environment("OMP_NUM_THREADS", str(overlay_policy.num_procs)):
                material_volumes = mesh.material_volumes(model, overlay_policy.n_samples)

            elements = [material_volumes.by_element(i) for i in range(material_volumes.num_elements)]
            with ThreadPoolExecutor(max_workers=overlay_policy.num_procs) as executor:
                materials = list(executor.map(
                    lambda element: Material.from_openmc_model_element(element,
                                                                       model,
                                                                       overlay_policy.mat_specs,
                                                                       overlay_policy.mix_policy),
                    elements))

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
        pitches = {key : relative_round(val, TOL) for key, val in self.pitch.items()}
        return hash((relative_round(self.xMin, TOL),
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

    def write_to_string(self, prefix: str = "", mpact_ids: Dict[PinMesh, int] = None) -> str:

        mpact_id = 1 if mpact_ids is None else mpact_ids[self]
        string = prefix
        string += f"pinmesh {mpact_id} gcyl {list_to_str(self._r_inside_bounds)} / " \
               +  f"{self._xMin} {self._xMax} {self._yMin} {self._yMax} / " \
               +  f"{list_to_str(self._zvals)} / {list_to_str(self._ndivr_inside_bounds)} / " \
               +  f"{list_to_str(self._ndiva_inside_bounds)} / {list_to_str(self._ndivz)}\n"

        return string

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

        def box_encloses_circle(r):
            return ((self.xMin < -r or isclose(self.xMin, -r)) and (r < self.xMax or isclose(self.xMax, r)) and
                    (self.yMin < -r or isclose(self.yMin, -r)) and (r < self.yMax or isclose(self.yMax, r)))

        def box_overlaps_circle(r):
            return not all(r < corner or isclose(r, corner) for corner in corners) or box_encloses_circle(r)

        radii_inside_bounds = [i for i,r in enumerate(self.r)
                               if box_overlaps_circle(r) and not circle_encloses_box(r)]

        assert radii_inside_bounds or circle_encloses_box(self.r[0]), \
            f"GCYL PinMesh with bounds {self.xMin, self.yMin, self.xMax, self.yMax} "+ \
             "has no cylinder regions which fall within the bounds"

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
                model:          openmc.Model,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy(),
    ) -> List[Optional[Material]]:

        raise NotImplementedError("Overlay not implemented for GeneralCylindricalPinMesh")
