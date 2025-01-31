from __future__ import annotations
from typing import Dict, List, Any
from math import isclose

from mpactpy.pin import Pin
from mpactpy.utils import list_to_str, is_rectangular, unique


class Module():
    """  Module of an MPACT model

    Attributes
    ----------
    nx : int
        Number of pins along the x-dimension
    ny : int
        Number of pins along the y-dimension
    nz : int
        Number of pins along the z-dimension
    pitch : Dict[str, float]
        The pitch of the module in each axis direction (keys: 'X', 'Y', 'Z') (cm)
    pin_map : List[List[Pin]]
        The 2-D array of pin.  This array is extruded
        nz times in the z-direction
    pins : List[Pin]
        The unique pins of this module
    pinmeshes : List[PinMesh]
        The unique pin meshes of this module
    materials : List[Material]
        The materials of this module
    """

    @property
    def nx(self) -> int:
        return len(self.pin_map)

    @property
    def ny(self) -> int:
        return  len(self.pin_map[0])

    @property
    def nz(self) -> int:
        return self._nz

    @property
    def pitch(self) -> Dict[str, float]:
        return self._pitch

    @property
    def pin_map(self) -> List[List[Pin]]:
        return self._pin_map

    def __init__(self, nz: int, pin_map: List[List[Pin]]):

        assert nz > 0
        assert is_rectangular(pin_map)

        self._nz      = nz
        self._pin_map = pin_map

        assert all(isclose(self.pin_map[i][j].pitch['X'], self.pin_map[0][j].pitch['X'])
                   for j in range(self.ny) for i in range(self.nx))

        assert all(isclose(self.pin_map[i][j].pitch['Y'], self.pin_map[i][0].pitch['Y'])
                   for i in range(self.nx) for j in range(self.ny))

        assert all(isclose(pin.pitch['Z'], self.pin_map[0][0].pitch['Z'])
                   for row in self.pin_map for pin in row)

        x_pitch     = sum(self.pin_map[0][j].pitch['X'] for j in range(self.ny))
        y_pitch     = sum(self.pin_map[i][0].pitch['Y'] for i in range(self.nx))
        z_pitch     = self.pin_map[0][0].pitch['Z'] * self.nz
        self._pitch = {'X': x_pitch, 'Y': y_pitch, 'Z': z_pitch}


    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Module)     and
                self.nz      == other.nz      and
                self.pin_map == other.pin_map
               )

    def __hash__(self) -> int:
        return hash((self.nz,
                     tuple(tuple(row) for row in self.pin_map)))

    def write_to_string(self,
                        prefix: str = "",
                        pin_mpact_ids: Dict[Pin, int] = None,
                        module_mpact_ids: Dict[Module, int] = None) -> str:
        """ Method for writing a module to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string
        pin_mpact_ids : Dict[Pin, int]
            A collection of Pins and their corresponding MPACT IDs
        module_mpact_ids : Dict[Pin, int]
            A collection of Modules and their corresponding MPACT IDs

        Returns
        -------
        str
            The string that represents the module
        """

        if pin_mpact_ids is None:
            pins = unique(pin for row in self.pin_map for pin in row)
            pin_mpact_ids = {pin: i for i, pin in enumerate(pins)}

        module_id = 1 if module_mpact_ids is None else module_mpact_ids[self]

        id_length = max(len(str(pin_mpact_ids[pin])) for row in self.pin_map for pin in row)

        string = prefix + f"module {module_id} {self.nx} {self.ny} {self.nz}\n"
        for row in self.pin_map:
            pins = [pin_mpact_ids[pin] for pin in row]
            string += prefix + prefix + f"{list_to_str(pins, id_length)}\n"
        return string


    def get_axial_slice(self, start_pos: float, stop_pos: float) -> Module:
        """ Method for creating a new Module from an axial slice of this Module

        Parameters
        ----------
        start_pos : float
            The starting axial position of the slice
        stop_pos : float
            The stopping axial position of the slice

        Returns
        -------
        Module
            The new Module created from the axial slice
        """

        assert stop_pos > start_pos

        if stop_pos  <= 0.              or isclose(stop_pos,  0.) or \
           start_pos >= self.pitch["Z"] or isclose(start_pos, self.pitch["Z"]):
            return None

        start_pos = max(0,               start_pos)
        stop_pos  = min(self.pitch["Z"], stop_pos)

        pin_map = []
        for row in self.pin_map:
            pin_map.append([])
            for pin in row:
                stack = None
                for z in range(self.nz):
                    z0 = z * pin.pitch["Z"]
                    pin_slice = pin.get_axial_slice(start_pos-z0, stop_pos-z0)
                    if pin_slice:
                        stack = stack.axial_merge(pin_slice) if stack else pin_slice
                pin_map[-1].append(stack)

        return Module(1, pin_map)
