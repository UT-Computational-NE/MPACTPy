from __future__ import annotations
from typing import Dict, List, Any, TypedDict
from math import isclose

from mpactpy.module import Module
from mpactpy.utils import list_to_str, is_rectangular, unique


class Lattice():
    """  Lattice of an MPACT model

    Attributes
    ----------
    nx : int
        Number of modules along the x-dimension
    ny : int
        Number of modules along the y-dimension
    pitch : Pitch
        The pitch of the lattice in each axis direction (keys: 'X', 'Y', 'Z') (cm)
    mod_dim : ModDim
        The x,y dimensions of the ray-tracing module
    module_map : List[List[Module]]
        a 2-D array of modules
    """

    class Pitch(TypedDict):
        """ A Typed Dictionary class for Lattice Pitches
        """
        X: float
        Y: float
        Z: float

    class ModDim(TypedDict):
            """ A Typed Dictionary class for Lattice Module Dimensions
            """
            X: float
            Y: float
            Z: float

    @property
    def nx(self) -> int:
        return len(self.module_map)

    @property
    def ny(self) -> int:
        return len(self.module_map[0])

    @property
    def pitch(self) -> Pitch:
        return self._pitch

    @property
    def mod_dim(self) -> ModDim:
        return self._mod_dim

    @property
    def module_map(self) -> List[List[Module]]:
        return self._module_map


    def __init__(self, module_map: List[List[Module]]):
        assert is_rectangular(module_map)

        self._module_map = module_map
        self._mod_dim    = {'X': self.module_map[0][0].pitch['X'],
                            'Y': self.module_map[0][0].pitch['Y']}

        assert all(isclose(self.module_map[i][j].pitch['X'], self.mod_dim['X'])
                   for j in range(self.ny) for i in range(self.nx))

        assert all(isclose(self.module_map[i][j].pitch['Y'], self.mod_dim['Y'])
                   for i in range(self.nx) for j in range(self.ny))

        assert all(isclose(module.pitch['Z'], self.module_map[0][0].pitch['Z'])
                   for row in self.module_map for module in row)

        self._pitch   = {'X': self.mod_dim['X'] * self.nx,
                         'Y': self.mod_dim['Y'] * self.ny,
                         'Z': self.module_map[0][0].pitch['Z']}

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Lattice)     and
                self.module_map == other.module_map
               )

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.module_map))

    def write_to_string(self,
                        prefix: str = "",
                        module_mpact_ids: Dict[Module, int] = None,
                        lattice_mpact_ids: Dict[Lattice, int] = None) -> str:
        """ Method for writing a lattice to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string
        module_mpact_ids : Dict[Module, int]
            A collection of Modules and their corresponding MPACT IDs
        lattice_mpact_ids : Dict[Lattice, int]
            A collection of Lattices and their corresponding MPACT IDs

        Returns
        -------
        str
            The string that represents the lattice
        """

        if module_mpact_ids is None:
            modules = unique(module for row in self.module_map for module in row)
            module_mpact_ids = {module: i for i, module in enumerate(modules)}

        lattice_id = 1 if lattice_mpact_ids is None else lattice_mpact_ids[self]

        id_length = max(len(str(module_mpact_ids[module])) for row in self.module_map for module in row)

        string = prefix + f"lattice {lattice_id} {self.nx} {self.ny}\n"
        for row in self.module_map:
            modules = [module_mpact_ids[module] for module in row]
            string += prefix + prefix + f"{list_to_str(modules, id_length)}\n"
        return string

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
