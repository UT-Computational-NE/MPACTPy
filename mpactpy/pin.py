from __future__ import annotations
from typing import Dict, List, Any, Union, TypedDict, Tuple
from math import isclose, inf, pi
from copy import deepcopy
from itertools import accumulate

from mpactpy.material import Material
from mpactpy.pinmesh import PinMesh, RectangularPinMesh, GeneralCylindricalPinMesh
from mpactpy.utils import list_to_str, unique


class Pin():
    """ Pin of an MPACT model

    Attributes
    ----------
    pinmesh : PinMesh
        The pin mesh associated with this pin
    materials : List[Material]
        The materials used in each pin XSR. The number of
        entries must be equal to the number of uniform
        material regions defined in the pin mesh
    pitch : Pitch
        The pitch of the pin in each axis direction (keys: 'X', 'Y', 'Z') (cm)
    """

    class Pitch(TypedDict):
        """ A Typed Dictionary class for Pin Pitches
        """
        X: float
        Y: float
        Z: float

    @property
    def pinmesh(self) -> PinMesh:
        return self._pinmesh

    @property
    def materials(self) -> List[Material]:
        return self._materials

    @property
    def pitch(self) -> Pitch:
        return self._pinmesh.pitch

    def __init__(self, pinmesh: PinMesh, materials: List[Material]):
        assert(len(materials) == pinmesh.number_of_material_regions), \
            f"len(materials) = {len(materials)}, " + \
            f"pinmesh.number_of_material_regions = {pinmesh.number_of_material_regions}"

        self._pinmesh   = pinmesh
        self._materials = materials

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Pin)            and
                self.pinmesh   == other.pinmesh   and
                self.materials == other.materials
               )

    def __hash__(self) -> int:
        return hash((self.pinmesh,
                     tuple(self.materials)))

    def write_to_string(self,
                        prefix: str = "",
                        material_mpact_ids: Dict[Material, int] = None,
                        pinmesh_mpact_ids: Dict[PinMesh, int] = None,
                        pin_mpact_ids: Dict[Pin, int] = None) -> str:
        """ Method for writing a pin to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string
        material_mpact_ids : Dict[Material, int]
            A collection of Materials and their corresponding MPACT IDs
        pinmesh_mpact_ids : Dict[PinMesh, int]
            A collection of PinMeshes and their corresponding MPACT IDs
        pin_mpact_ids : Dict[Pin, int]
            A collection of Pins and their corresponding MPACT IDs

        Returns
        -------
        str
            The string that represents the pin
        """

        if material_mpact_ids is None:
            materials = unique(material for material in self.materials)
            material_mpact_ids = {material: i for i, material in enumerate(materials)}

        pinmesh_id = 1 if pinmesh_mpact_ids is None else pinmesh_mpact_ids[self.pinmesh]
        pin_id     = 1 if pin_mpact_ids     is None else pin_mpact_ids[self]

        materials = [material_mpact_ids[self.materials[i]] for i in self.pinmesh.regions_inside_bounds]
        string = prefix + f"pin {pin_id} {pinmesh_id} / {list_to_str(materials)}\n"
        return string

    def get_axial_slice(self, start_pos: float, stop_pos: float) -> Union[Pin, None]:
        """ Method for creating a new Pin from an axial slice of this Pin

        Parameters
        ----------
        start_pos : float
            The starting axial position of the slice
        stop_pos : float
            The stopping axial position of the slice

        Returns
        -------
        Union[Pin, None]
            The new Pin created from the axial slice, or None if the axial slice is outside the Pin bounds
        """

        assert stop_pos > start_pos, f"start_pos = {start_pos}, stop_pos = {stop_pos}"

        if stop_pos  <= 0.              or isclose(stop_pos,  0.) or \
           start_pos >= self.pitch["Z"] or isclose(start_pos, self.pitch["Z"]):
            return None

        start_pos = max(0,               start_pos)
        stop_pos  = min(self.pitch["Z"], stop_pos)

        zvals, ndivz, indices = [], [], []
        for i, (zval, ndiv) in enumerate(zip(self.pinmesh.zvals, self.pinmesh.ndivz)):
            if zval > start_pos or isclose(zval, start_pos):
                slice_zval = min(zval, stop_pos) - start_pos
                zvals.append(slice_zval)
                ndivz.append(max(1, int(ndiv*slice_zval/zval)))
                indices.append(i)
                if zval > stop_pos or isclose(zval, stop_pos):
                    break

        pinmesh = deepcopy(self.pinmesh)
        pinmesh.set_axial_mesh(zvals, ndivz)

        num_mats_per_axial = int((self.pinmesh.number_of_material_regions) / len(self.pinmesh.zvals))
        materials = [material for i in indices
                     for material in self.materials[(i)*num_mats_per_axial : (i)*num_mats_per_axial + num_mats_per_axial]]

        return Pin(pinmesh, materials)

    def axial_merge(self, pin: Pin) -> Pin:
        """ Method for axially merging a pin with this pin

        Parameters
        ----------
        pin : Pin
            The pin to axially merge with this pin

        Returns
        -------
        Pin
            The new combined pin
        """

        pinmesh_template = deepcopy(self.pinmesh)
        pinmesh_template.set_axial_mesh(pin.pinmesh.zvals, pin.pinmesh.ndivz)

        assert pin.pinmesh == pinmesh_template, "Pin meshes of the two pins must have the same radial specifications"

        zvals  = self.pinmesh.zvals[:]
        zvals += [zval + zvals[-1] for zval in pin.pinmesh.zvals]

        ndivz  = self.pinmesh.ndivz[:]
        ndivz += pin.pinmesh.ndivz[:]

        pinmesh_template.set_axial_mesh(zvals, ndivz)

        materials = self.materials + pin.materials
        return Pin(pinmesh_template, materials)


def build_rec_pin(thicknesses:             Dict[str, List[float]],
                  materials:               List[Material],
                  target_cell_thicknesses: Dict[str, float] = {}) -> Pin:
    """ Factory for building Rectangular Mesh Pins

    This factory, unlike construction via the "normal means" using
    RectangularPinMesh, allows users to specify a target material region thickness
    for each of the 3 axes.  This will make the subdivisions for each material region
    such that cell thickness will not exceed the target cell thicknesses.

    Parameters
    ----------
    thicknesses : Dict[str, List[float]]
        The thicknesses of each material region division in each axis-direction
        (keys: "X", "Y", "Z")
    materials : List[Material]
        The list of materials ordered from top-to-bottom and left-to-right for the material divisions
    target_cell_thicknesses : Dict[str, float]
        The target side length of the cells (cm).  Defaults to infinity for dimensions where no targets provided.
        (keys: "X", "Y", "Z")

    Returns
    -------
    Pin
        The constructed Pin
    """

    for dim in ["X", "Y", "Z"]:
        target_cell_thicknesses.setdefault(dim, inf)

    assert all(thickness > 0. for axis in thicknesses.values() for thickness in axis)
    assert all(thickness > 0. for thickness in target_cell_thicknesses.values())
    assert len(materials) == len(thicknesses["X"]) * len(thicknesses["Y"]) * len(thicknesses["Z"])
    ndiv = {axis: [max(1, int(thickness // target_cell_thicknesses[axis]))
                   for thickness in thicknesses[axis]] for axis in ["X", "Y", "Z"]}
    vals = {axis: list(accumulate(thicknesses[axis])) for axis in ["X", "Y", "Z"]}
    mesh  = RectangularPinMesh(vals["X"], vals["Y"], vals["Z"], ndiv["X"], ndiv["Y"], ndiv["Z"])
    return Pin(mesh, materials)



def build_gcyl_pin(bounds:                  Tuple[float, float, float, float],
                   thicknesses:             Dict[str, List[float]],
                   materials:               List[Material],
                   target_cell_thicknesses: Dict[str, float] = {}) -> Pin:
    """ Factory for building General Cylindrical Mesh Pins

    This factory, unlike construction via the "normal means" using
    GeneralCylindricalPinMesh, allows users to specify a target material region thickness
    for R, Arc-Length, and Z.  This will make the subdivisions for each material region
    such that cell thickness will not exceed the target cell thicknesses. For Arc-length,
    the outer most radius is considered when determining how much to subdivide,
    with the resultant number of azimuthal subdivisions being applied to all material regions.
    Also with arc-length, the number of subdivisions is rounded up to the nearest number divisible by 4
    since in most practical use cases, the number of azimuthal devisions should be evenly split over pin quadrants.

    Parameters
    ----------
    bounds: Tuple[float, float, float, float]
        The bounds of the GCYL pincell (order: x_min, x_max, y_min, y_max)
    thicknesses : Dict[str, List[float]]
        The thicknesses of each material region division in each axis-direction
        (keys: "R", "Z")
    materials : List[Material]
        The list of materials ordered from top-to-bottom and left-to-right for the material divisions
    target_cell_thicknesses : Dict[str, float]
        The target side length of the cells (cm). Defaults to infinity for dimensions where no targets provided.
        (keys: "R", "S", "Z")

    Returns
    -------
    Pin
        The constructed Pin
    """

    xmin = bounds[0]
    xmax = bounds[1]
    ymin = bounds[2]
    ymax = bounds[3]

    for dim in ["R", "S", "Z"]:
        target_cell_thicknesses.setdefault(dim, inf)

    assert xmin < xmax
    assert ymin < ymax
    assert all(thickness > 0. for axis in thicknesses.values() for thickness in axis)
    assert all(thickness > 0. for thickness in target_cell_thicknesses.values())
    assert len(materials) == (len(thicknesses['R']) + 1) * len(thicknesses['Z'])

    r_subds     = []
    num_r_subds = []
    for thickness in thicknesses["R"]:
        num_subd = max(1, int(thickness // target_cell_thicknesses["R"]))
        r_subds.extend([thickness / num_subd] * num_subd)
        num_r_subds.append(num_subd)

    r     = list(accumulate(r_subds))
    ndivr = [1] * len(r) # Subdivision was already performed by the target_cell_thickness

    # Special arithmetic here ensures azimuthal subdivisions are divisible by 4
    num_a_subd = max(1, (int(2.*pi*r[-1] // target_cell_thicknesses["S"]) + 3) // 4 * 4)
    ndiva      = [num_a_subd] * len(r) + [num_a_subd]

    ndivz = [max(1, int(thickness // target_cell_thicknesses["Z"])) for thickness in thicknesses["Z"]]
    zvals = list(accumulate(thicknesses["Z"]))

    subdivided_materials = []
    m = 0
    for _ in zvals:
        for num_subd in num_r_subds:
            subdivided_materials.extend([materials[m]] * num_subd)
            m += 1
        subdivided_materials.append(materials[m])
        m += 1

    mesh = GeneralCylindricalPinMesh(r, xmin, xmax, ymin, ymax, zvals, ndivr, ndiva, ndivz)
    return Pin(mesh, subdivided_materials)
