from __future__ import annotations
from typing import Dict, List, Any
from math import isclose

from mpactpy.material import Material
from mpactpy.pinmesh import PinMesh
from mpactpy.pin import Pin
from mpactpy.module import Module
from mpactpy.utils import list_to_str, unique, is_rectangular


class Lattice():
    """  Lattice of an MPACT model

    Attributes
    ----------
    id : int
        The ID of the module
    nx : int
        Number of modules along the x-dimension
    ny : int
        Number of modules along the y-dimension
    pitch : Dict[str, float]
        The pitch of the lattice in each axis direction (keys: 'X', 'Y', 'Z') (cm)
    mod_dim : Dict[str, float]
        The x,y dimensions of the ray-tracing module
    module_map : List[List[Module]]
        a 2-D array of modules
    modules : List[Module]
        The unique modules of this lattice
    pins : List[Pin]
        The unique pins of this lattice
    pinmeshes : List[PinMesh]
        The unique pin meshes of this lattice
    materials : List[Material]
        The materials of this module
    """

    @property
    def mpact_id(self) -> int:
        return self._mpact_id

    @mpact_id.setter
    def mpact_id(self, mpact_id: int) -> None:
        assert(mpact_id > 0), f"mpact_id = {mpact_id}"
        self._mpact_id = mpact_id

    @property
    def nx(self) -> int:
        return len(self.module_map)

    @property
    def ny(self) -> int:
        return len(self.module_map[0])

    @property
    def pitch(self) -> Dict[str, float]:
        return self._pitch

    @property
    def mod_dim(self) -> Dict[str, float]:
        return self._mod_dim

    @property
    def module_map(self) -> List[List[Module]]:
        return self._module_map

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

    def __init__(self, module_map: List[List[Module]], mpact_id: int = 1):
        assert is_rectangular(module_map)

        self.mpact_id    = mpact_id
        self._module_map = module_map
        self.set_unique_elements()
        self._mod_dim    = {'X': self.modules[0].pitch['X'], 'Y': self.modules[0].pitch['Y']}

        assert all(isclose(self.module_map[i][j].pitch['X'], self.mod_dim['X'])
                   for j in range(self.ny) for i in range(self.nx))

        assert all(isclose(self.module_map[i][j].pitch['Y'], self.mod_dim['Y'])
                   for i in range(self.nx) for j in range(self.ny))

        assert all(isclose(module.pitch['Z'], self.modules[0].pitch['Z']) for module in self.modules)

        self._pitch   = {'X': self.mod_dim['X'] * self.nx,
                         'Y': self.mod_dim['Y'] * self.ny,
                         'Z': self.modules[0].pitch['Z']}

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Lattice)     and
                self.module_map == other.module_map
               )

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.module_map))

    def write_to_string(self, prefix: str = "") -> str:
        """ Method for writing a lattice to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string

        Returns
        -------
        str
            The string that represents the lattice
        """

        id_length = max(len(str(module.mpact_id)) for row in self.module_map for module in row)

        string = prefix + f"lattice {self.mpact_id} {self.nx} {self.ny}\n"
        for row in self.module_map:
            modules = [module.mpact_id for module in row]
            string += prefix + prefix + f"{list_to_str(modules, id_length)}\n"
        return string

    def set_unique_elements(self, other_modules: List[Module] = []) -> None:
        """ Determines and sets the unique elements of the lattice

        Parameters
        ----------
        other_modules : List[Modules]
            The modules from other lattices which should be considered as already defined
        """
        # NOTE: To get the ordering correct, other_modules must by left-hand-side added to the map list
        already_defined_modules = other_modules + [module for row in self.module_map for module in row]
        already_defined_modules = {module: i+1 for i, module in enumerate(unique(already_defined_modules))}
        for module, mpact_id in already_defined_modules.items():
            module.mpact_id = mpact_id

        self._modules   = list(already_defined_modules)
        self._pins      = unique([pin for module in self.modules for row in module.pin_map for pin in row])
        self._pinmeshes = unique([pin.pinmesh for pin in self.pins])
        self._materials = unique([material for pin in self.pins for material in pin.materials])
        for module in self.modules:
            module.set_unique_elements(self.pins)

        for i, _ in enumerate(self.module_map):
            for j, _ in enumerate(self.module_map[i]):
                self._module_map[i][j] = next(module for module in self.modules if self.module_map[i][j] == module)


    def get_axial_slice(self, start_pos: float, stop_pos: float) -> Lattice:
        """ Method for creating a new Lattice from an axial slice of this Lattice

        Parameters
        ----------
        start_pos : float
            The starting axial position of the slice
        stop_pos : float
            The stopping axial position of the slice

        Returns
        -------
        Lattice
            The new Lattice created from the axial slice
        """

        assert stop_pos > start_pos

        if stop_pos  <= 0.              or isclose(stop_pos,  0.) or \
           start_pos >= self.pitch["Z"] or isclose(start_pos, self.pitch["Z"]):
            return None

        start_pos = max(0,               start_pos)
        stop_pos  = min(self.pitch["Z"], stop_pos)

        module_map = [[module.get_axial_slice(start_pos, stop_pos) for module in row] for row in self.module_map]

        return Lattice(module_map)
