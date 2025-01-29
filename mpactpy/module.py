from __future__ import annotations
from typing import Dict, List, Any
from math import isclose

from mpactpy.material import Material
from mpactpy.pinmesh import PinMesh
from mpactpy.pin import Pin
from mpactpy.utils import list_to_str, unique, is_rectangular


class Module():
    """  Module of an MPACT model

    Attributes
    ----------
    mpact_id : int
        The MPACT ID of the module
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
    def mpact_id(self) -> int:
        return self._mpact_id

    @mpact_id.setter
    def mpact_id(self, mpact_id: int) -> None:
        assert(mpact_id > 0), f"mpact_id = {mpact_id}"
        self._mpact_id = mpact_id

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

    @property
    def pins(self) -> List[Pin]:
        return self._pins

    @property
    def pinmeshes(self) -> List[PinMesh]:
        return self._pinmeshes

    @property
    def materials(self) -> List[Material]:
        return self._materials

    def __init__(self, nz: int, pin_map: List[List[Pin]], mpact_id: int = 1):

        assert nz > 0
        assert is_rectangular(pin_map)

        self.mpact_id = mpact_id
        self._nz      = nz
        self._pin_map = pin_map
        self.set_unique_elements()

        assert all(isclose(self.pin_map[i][j].pitch['X'], self.pin_map[0][j].pitch['X'])
                   for j in range(self.ny) for i in range(self.nx))

        assert all(isclose(self.pin_map[i][j].pitch['Y'], self.pin_map[i][0].pitch['Y'])
                   for i in range(self.nx) for j in range(self.ny))

        assert all(isclose(pin.pitch['Z'], self.pins[0].pitch['Z']) for pin in self.pins)

        x_pitch     = sum(self.pin_map[0][j].pitch['X'] for j in range(self.ny))
        y_pitch     = sum(self.pin_map[i][0].pitch['Y'] for i in range(self.nx))
        z_pitch     = self.pins[0].pitch['Z'] * self.nz
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

    def write_to_string(self, prefix: str = "") -> str:
        """ Method for writing a module to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string

        Returns
        -------
        str
            The string that represents the module
        """

        id_length = max(len(str(pin.mpact_id)) for row in self.pin_map for pin in row)

        string = prefix + f"module {self.mpact_id} {self.nx} {self.ny} {self.nz}\n"
        for row in self.pin_map:
            pins = [pin.mpact_id for pin in row]
            string += prefix + prefix + f"{list_to_str(pins, id_length)}\n"
        return string


    def set_unique_elements(self, other_pins: List[Pin]  = []) -> None:
        """ Determines and sets the unique elements of the module

        Parameters
        ----------
        other_pins : List[Pin]
            The pins from other modules which should be considered as already defined
        """
        # NOTE: To get the ordering correct, other_pins must by left-hand-side added to the map list
        already_defined_pins      = unique(other_pins + [pin for row in self.pin_map for pin in row])
        already_defined_pinmeshes = unique([pin.pinmesh for pin in already_defined_pins])
        already_defined_materials = unique([material for pin in already_defined_pins for material in pin.materials])

        for i, _ in enumerate(self.pin_map):
            for j, _ in enumerate(self.pin_map[i]):
                self.pin_map[i][j].set_unique_elements(already_defined_pinmeshes, already_defined_materials)
                self._pin_map[i][j] = next(pin for pin in already_defined_pins if self.pin_map[i][j] == pin)

        self._pins      = unique([pin for row in self.pin_map for pin in row])
        self._pinmeshes = unique([pin.pinmesh for pin in self.pins])
        self._materials = unique([material for pin in self.pins for material in pin.materials])


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
