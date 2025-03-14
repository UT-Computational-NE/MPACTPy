from __future__ import annotations
from typing import Dict, List, Any, TypedDict
from math import isclose

from mpactpy.lattice import Lattice
from mpactpy.utils import list_to_str, unique


class Assembly():
    """ Assembly of an MPACT model

    Parameters
    ----------
    lattice_map : List[Lattice]
        1-D array of lattice names

    Attributes
    ----------
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
    """

    class ModDim(TypedDict):
        """ A Typed Dictionary class for Assembly Module Dimensions
        """
        X: float
        Y: float
        Z: List[float]

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


    def __init__(self, lattice_map: List[Lattice]):
        assert len(lattice_map) > 0

        self._lattice_map = lattice_map

        assert all(isclose(lattice.mod_dim['X'], self.lattice_map[0].mod_dim['X']) for lattice in self.lattice_map)
        assert all(isclose(lattice.mod_dim['Y'], self.lattice_map[0].mod_dim['Y']) for lattice in self.lattice_map)
        assert all(lattice.nx == self.lattice_map[0].nx for lattice in self.lattice_map)
        assert all(lattice.ny == self.lattice_map[0].ny for lattice in self.lattice_map)

        self._pitch  = {'X': self.lattice_map[0].pitch['X'],
                        'Y': self.lattice_map[0].pitch['Y']}

        self._height = sum(self.lattice_map[i].pitch['Z'] for i in range(self.nz))

        unique_lattice_heights = []
        for lattice in self.lattice_map:
            if not any(isclose(lattice.pitch['Z'], unique_height) for unique_height in unique_lattice_heights):
                unique_lattice_heights.append(lattice.pitch['Z'])
        unique_lattice_heights = sorted(unique_lattice_heights)

        self._mod_dim = {'X': self.lattice_map[0].mod_dim['X'],
                         'Y': self.lattice_map[0].mod_dim['Y'],
                         'Z': unique_lattice_heights}

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Assembly)     and
                self.lattice_map == other.lattice_map
               )

    def __hash__(self) -> int:
        return hash(tuple(self.lattice_map))

    def write_to_string(self, prefix: str = "",
                        lattice_mpact_ids: Dict[Lattice, int] = None,
                        assembly_mpact_ids: Dict[Assembly, int] = None) -> str:
        """ Method for writing a to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string
        lattice_mpact_ids : Dict[Pin, int]
            A collection of Lattices and their corresponding MPACT IDs
        assembly_mpact_ids : Dict[Pin, int]
            A collection of Assemblies and their corresponding MPACT IDs

        Returns
        -------
        str
            The string that represents the
        """

        if lattice_mpact_ids is None:
            lattice = unique(lattice for lattice in self.lattice_map)
            lattice_mpact_ids = {lattice: i+1 for i, lattice in enumerate(lattice)}

        assembly_id = 1 if assembly_mpact_ids is None else assembly_mpact_ids[self]

        string = prefix + f"assembly {assembly_id}\n"
        lattices = [lattice_mpact_ids[lattice] for lattice in self.lattice_map]
        string += prefix + prefix + f"{list_to_str(lattices)}\n"
        return string


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


    def with_height(self, height: float) -> Assembly:
        """ Method for changing the height of 2D Assemblies

        Parameters
        ----------
        height : float
            The height of the new 2D assembly

        Returns
        -------
        Assembly
            The new assembly with the new height
        """

        assert self.nz == 1, f"nz = {self.nz}, Assembly must be strictly 2D"

        return Assembly([self.lattice_map[0].with_height(height)])
