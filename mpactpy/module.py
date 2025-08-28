from __future__ import annotations
from typing import Dict, List, Any, TypedDict, Tuple, Optional
from math import isclose
from concurrent.futures import ProcessPoolExecutor

import openmc

from mpactpy.pin import Pin, PinMesh
from mpactpy.material import Material
from mpactpy.utils import list_to_str, is_rectangular, unique


class Module():
    """  Module of an MPACT model

    Parameters
    ----------
    nz : int
        Number of pins along the z-dimension
    pin_map : List[List[Pin]]
        The 2-D array of pin.  This array is extruded
        nz times in the z-direction

    Attributes
    ----------
    nx : int
        Number of pins along the x-dimension
    ny : int
        Number of pins along the y-dimension
    nz : int
        Number of pins along the z-dimension
    pitch : Pitch
        The pitch of the module in each axis direction (keys: 'X', 'Y', 'Z') (cm)
    pin_map : List[List[Pin]]
        The 2-D array of pin.  This array is extruded
        nz times in the z-direction
    pins : List[Pin]
        The unique pins contained in this module
    pinmeshes : List[PinMesh]
        The unique pinmeshes contained in this module
    materials : List[Material]
        The unique materials contained in this module
    """

    class Pitch(TypedDict):
        """ A Typed Dictionary class for Module Pitches
        """
        X: float
        Y: float
        Z: float

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
    def pitch(self) -> Pitch:
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

        self._pins      = unique([pin for row in self.pin_map for pin in row])
        self._pinmeshes = unique([pin.pinmesh for pin in self.pins])
        self._materials = unique([material for pin in self.pins for material in pin.unique_materials])


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
            pin_mpact_ids = {pin: i+1 for i, pin in enumerate(pins)}

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


    def with_height(self, height: float) -> Module:
        """ Method for changing the height of 2D Modules

        Parameters
        ----------
        height : float
            The height of the new 2D module

        Returns
        -------
        Module
            The new module with the new height
        """

        assert self.nz == 1, f"nz = {self.nz}, Module must be strictly 2D"

        return Module(1, [[pin.with_height(height) for pin in row]
                      for row in self.pin_map])


    OverlayMask = Dict[Pin, Optional[Pin.OverlayMask]]

    def has_overlay_work(self, include_only: Optional[OverlayMask] = None) -> bool:
        """Check if this module has actual overlay work to do based on the include mask.

        Parameters
        ----------
        include_only : Optional[OverlayMask]
            The dictionary of pins and their masks to include for this module

        Returns
        -------
        bool
            True if module has overlay work to do, False otherwise
        """
        if include_only is None:
            # No mask means include all pins in this module
            return True

        # Check if module contains any pins that have overlay work to do
        for pin in self.pins:
            if pin in include_only:
                if pin.has_overlay_work(include_only[pin]):
                    return True

        return False

    def overlay(self,
                geometry:       openmc.Geometry,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                include_only:   Optional[OverlayMask] = None,
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy()) -> Module:
        """ A method for overlaying an OpenMC geometry over top an MPACTPy Module

        Parameters
        ----------
        geometry : openmc.Geometry
            The OpenMC Geometry to be mapped onto the MPACTPy Module
        offset : Tuple(float, float, float)
            Offset of the OpenMC geometry's lower-left corner relative to the
            MPACT Module lower-left. Default is (0.0, 0.0, 0.0)
        include_only : Optional[OverlayMask]
            Specifies which MPACT elements should be considered during overlay.
            If None, all elements are included.
        overlay_policy : OverlayPolicy
            A configuration object specifying how a mesh overlay should be done.

        Returns
        -------
        Module
            A new MPACTPy Module which is a copy of the original,
            but with the OpenMC Geometry overlaid on top.
        """

        include_only: Module.OverlayMask = include_only if include_only else \
                                           {pin: None for row in self.pin_map for pin in row}

        x0, y0, z0 = offset

        # Collect all pin work to be done
        pin_work = []
        y = y0 + self.pitch['Y']
        for i, row in enumerate(self.pin_map):
            x = x0
            y -= row[0].pitch['Y']
            for j, pin in enumerate(row):
                if pin in include_only:
                    if pin.has_overlay_work(include_only[pin]):
                        pin_work.append((pin, (x, y, z0), include_only[pin], i, j))
                x += pin.pitch['X']

        # Determine parallelization
        num_pins         = len(pin_work)
        num_module_procs = min(num_pins, overlay_policy.num_procs)
        child_policy     = overlay_policy.allocate_processes(num_pins)

        # Process pins in parallel
        with ProcessPoolExecutor(max_workers=num_module_procs) as executor:
            futures = [
                executor.submit(self._overlay_pin_worker, pin, offset_pos, include_mask, geometry, child_policy)
                for pin, offset_pos, include_mask, _, _ in pin_work
            ]
            overlaid_pins = [future.result() for future in futures]

        # Reconstruct the pin map with overlaid pins
        new_pin_map = [row[:] for row in self.pin_map]
        for (_, _, _, i, j), overlaid in zip(pin_work, overlaid_pins):
            new_pin_map[i][j] = overlaid

        return Module(1, new_pin_map)

    @staticmethod
    def _overlay_pin_worker(pin:            Pin,
                            offset:         Tuple[float, float, float],
                            include_mask:   Optional[Pin.OverlayMask],
                            geometry:       openmc.Geometry,
                            overlay_policy: PinMesh.OverlayPolicy) -> Pin:
        """Worker function for parallel pin overlay processing.

        Parameters
        ----------
        pin : Pin
            The MPACTPy Pin to overlay with the OpenMC geometry.
        offset : Tuple[float, float, float]
            The (x, y, z) offset coordinates for the OpenMC geometry relative to
            the pin's lower-left corner.
        include_mask : Optional[Pin.OverlayMask]
            Optional mask specifying which elements within the pin should be
            included in the overlay operation. If None, all pin elements are included.
        geometry : openmc.Geometry
            The OpenMC Geometry to be overlaid onto the pin.
        overlay_policy : PinMesh.OverlayPolicy
            Configuration object specifying overlay method, sampling parameters,
            and process allocation for cascading parallelization.

        Returns
        -------
        Pin
            A new Pin instance with the OpenMC geometry overlaid.
        """
        return pin.overlay(geometry, offset, include_mask, overlay_policy)
