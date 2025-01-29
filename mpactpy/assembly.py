from __future__ import annotations
from typing import Dict, List, Any, TypedDict
from math import isclose

from mpactpy.material import Material
from mpactpy.pinmesh import PinMesh
from mpactpy.pin import Pin
from mpactpy.module import Module
from mpactpy.lattice import Lattice
from mpactpy.utils import list_to_str, unique


class Assembly():
    """ Assembly of an MPACT model

    Attributes
    ----------
    id : int
        The ID of the module
    pitch : Dict[str, float]
        The pitch of the lattice in the X-Y axis direction (keys: 'X', 'Y') (cm)
    height : float
        The total height of the assembly (cm)
    nz : int
        Number of modules along the z-dimension
    mod_dim : ModDim
        The x,y,z dimensions of the ray-tracing module
    lattice_map : List[Lattice]
        1-D array of lattice names
    lattices : List[Lattice]
        The unique lattices of this assembly
    modules : List[Module]
        The unique modules of this assembly
    pins : List[Pin]
        The unique pins of this assembly
    pinmeshes : List[PinMesh]
        The unique pin meshes of this assembly
    materials : List[Material]
        The materials of this module
    """

    class ModDim(TypedDict):
        """ A Typed Dictionary class for Assembly Module Dimensions
        """
        X: float
        Y: float
        Z: List[float]

    @property
    def mpact_id(self) -> int:
        return self._mpact_id

    @mpact_id.setter
    def mpact_id(self, mpact_id: int) -> None:
        assert(mpact_id > 0), f"mpact_id = {mpact_id}"
        self._mpact_id = mpact_id

    @property
    def pitch(self) -> Dict[str, float]:
        return {'X': self.lattice_map[0].pitch['X'], 'Y': self.lattice_map[0].pitch['Y']}

    @property
    def height(self) -> float:
        return self._height

    @property
    def nz(self) -> int:
        return len(self.lattice_map)

    @property
    def mod_dim(self) -> ModDim:
        return self._mod_dim

    @property
    def lattice_map(self) -> List[Lattice]:
        return self._lattice_map

    @property
    def lattices(self) -> List[Lattice]:
        return self._lattices

    @property
    def modules(self) -> List[Module]:
        return self._modules

    @property
    def pins(self) -> List[Pin]:
        return self._pins

    @property
    def pinmeshes(self) -> List[PinMesh]:
        return self._pinmeshes

    @property
    def materials(self) -> List[Material]:
        return self._materials

    def __init__(self, lattice_map: List[Lattice], mpact_id: int = 1):
        assert len(lattice_map) > 0

        self.mpact_id     = mpact_id
        self._lattice_map = lattice_map
        self.set_unique_elements()

        assert all(isclose(lattice.mod_dim['X'], self.lattices[0].mod_dim['X']) for lattice in self.lattices)
        assert all(isclose(lattice.mod_dim['Y'], self.lattices[0].mod_dim['Y']) for lattice in self.lattices)
        assert all(lattice.nx == self.lattices[0].nx for lattice in self.lattices)
        assert all(lattice.ny == self.lattices[0].ny for lattice in self.lattices)

        self._pitch  = {'X': self.lattices[0].pitch['X'],
                        'Y': self.lattices[0].pitch['Y']}

        self._height = sum(self.lattice_map[i].pitch['Z'] for i in range(self.nz))

        unique_lattice_heights = []
        for lattice in self.lattices:
            if not any(isclose(lattice.pitch['Z'], unique_height) for unique_height in unique_lattice_heights):
                unique_lattice_heights.append(lattice.pitch['Z'])
        unique_lattice_heights = sorted(unique_lattice_heights)

        self._mod_dim = {'X': self.lattices[0].mod_dim['X'],
                         'Y': self.lattices[0].mod_dim['Y'],
                         'Z': unique_lattice_heights}

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Assembly)     and
                self.lattice_map == other.lattice_map
               )

    def __hash__(self) -> int:
        return hash(tuple(self.lattice_map))

    def write_to_string(self, prefix: str = "") -> str:
        """ Method for writing a to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string

        Returns
        -------
        str
            The string that represents the
        """

        string = prefix + f"assembly {self.mpact_id}\n"
        lattices = [lattice.mpact_id for lattice in self.lattice_map]
        string += prefix + prefix + f"{list_to_str(lattices)}\n"
        return string

    def set_unique_elements(self, other_lattices: List[Lattice] = []) -> None:
        """ Determines and sets the unique elements of the assembly

        Parameters
        ----------
        other_lattices : List[Lattice]
            The lattices from other assemblies which should be considered as already defined
        """
        # NOTE: To get the ordering correct, other_lattices must by left-hand-side added to the map list
        already_defined_lattices  = unique(other_lattices + self.lattice_map)
        already_defined_modules   = unique([module for lattice in already_defined_lattices for module in lattice.modules])

        for i, _ in enumerate(self.lattice_map):
            self.lattice_map[i].set_unique_elements(already_defined_modules)
            self._lattice_map[i] = next(lattice for lattice in already_defined_lattices if self.lattice_map[i] == lattice)

        self._lattices  = unique(self.lattice_map)
        self._modules   = unique([module for lattice in self.lattices for module in lattice.modules])
        self._pins      = unique([pin for module in self.modules for pin in module.pins])
        self._pinmeshes = unique([pin.pinmesh for pin in self.pins])
        self._materials = unique([material for pin in self.pins for material in pin.materials])


    def get_axial_slice(self, start_pos: float, stop_pos: float) -> Assembly:
        """ Method for creating a new Assembly from an axial slice of this Assembly

        Parameters
        ----------
        start_pos : float
            The starting axial position of the slice
        stop_pos : float
            The stopping axial position of the slice

        Returns
        -------
        Assembly
            The new Assembly created from the axial slice
        """

        assert stop_pos > start_pos

        if stop_pos  <= 0.          or isclose(stop_pos,  0.) or \
           start_pos >= self.height or isclose(start_pos, self.height):
            return None

        start_pos = max(0,           start_pos)
        stop_pos  = min(self.height, stop_pos)

        z0 = 0.
        lattice_map = []
        for lattice in self.lattice_map:
            lattice_slice = lattice.get_axial_slice(start_pos-z0, stop_pos-z0)
            if lattice_slice:
                lattice_map.append(lattice_slice)
            z0 += lattice.pitch["Z"]

        return Assembly(lattice_map)
