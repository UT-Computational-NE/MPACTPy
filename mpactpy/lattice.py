from __future__ import annotations
from typing import Dict, List, Any, TypedDict, Tuple, Optional
from math import isclose
from concurrent.futures import ProcessPoolExecutor, as_completed

import openmc

from mpactpy.module import PinMesh, Module
from mpactpy.pin import Pin
from mpactpy.material import Material
from mpactpy.utils import list_to_str, is_rectangular, unique


class Lattice():
    """  Lattice of an MPACT model

    Parameters
    ----------
    module_map : List[List[Module]]
        a 2-D array of modules

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
    modules : List[Module]
        The unique modules contained in this lattice
    pins : List[Pin]
        The unique pins contained in this lattice
    pinmeshes : List[PinMesh]
        The unique pinmeshes contained in this lattice
    materials : List[Material]
        The unique materials contained in this lattice
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

        self._modules   = unique([module for row in self.module_map for module in row])
        self._pins      = unique([pin for module in self.modules for pin in module.pins])
        self._pinmeshes = unique([pin.pinmesh for pin in self.pins])
        self._materials = unique([material for pin in self.pins for material in pin.unique_materials])

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
            module_mpact_ids = {module: i+1 for i, module in enumerate(modules)}

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


    def with_height(self, height: float) -> Lattice:
        """ Method for changing the height of 2D Lattices

        Parameters
        ----------
        height : float
            The height of the new 2D lattice

        Returns
        -------
        Lattice
            The new lattice with the new height
        """

        return Lattice([[module.with_height(height) for module in row]
                      for row in self.module_map])


    OverlayMask = Dict[Module, Optional[Module.OverlayMask]]

    def has_overlay_work(self, include_only: Optional[OverlayMask] = None) -> bool:
        """Check if this lattice has actual overlay work to do based on the include mask.

        Parameters
        ----------
        include_only : Optional[OverlayMask]
            The dictionary of modules and their masks to include for this lattice

        Returns
        -------
        bool
            True if lattice has overlay work to do, False otherwise
        """
        if include_only is None:
            # No mask means include all modules in this lattice
            return True

        # Check if lattice contains any modules that have overlay work to do
        for module in self.modules:
            if module in include_only:
                if module.has_overlay_work(include_only[module]):
                    return True

        return False

    def overlay(self,
                geometry:       openmc.Geometry,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                include_only:   Optional[OverlayMask] = None,
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy()) -> Lattice:
        """ A method for overlaying an OpenMC geometry over top an MPACTPy Lattice

        Parameters
        ----------
        geometry : openmc.Geometry
            The OpenMC Geometry to be mapped onto the MPACTPy Lattice
        offset : Tuple(float, float, float)
            Offset of the OpenMC geometry's lower-left corner relative to the
            MPACT Lattice lower-left. Default is (0.0, 0.0, 0.0)
        include_only : Optional[OverlayMask]
            Specifies which MPACT elements should be considered during overlay.
            If None, all elements are included.
        overlay_policy : OverlayPolicy
            A configuration object specifying how a mesh overlay should be done.

        Returns
        -------
        Lattice
            A new MPACTPy Lattice which is a copy of the original,
            but with the OpenMC Geometry overlaid on top.
        """

        include_only: Lattice.OverlayMask = include_only if include_only else \
                                            {module: None for row in self.module_map for module in row}

        x0, y0, z0 = offset

        # Collect all module work to be done
        module_work = []
        y = y0 + self.pitch['Y']
        for i, row in enumerate(self.module_map):
            x = x0
            y -= row[0].pitch['Y']
            for j, module in enumerate(row):
                if module in include_only:
                    if module.has_overlay_work(include_only[module]):
                        module_work.append((module, (x, y, z0), include_only[module], i, j))
                x += module.pitch['X']

        # Determine parallelization strategy
        num_modules       = len(module_work)
        num_lattice_procs = min(num_modules, overlay_policy.num_procs)
        child_policy      = overlay_policy.allocate_processes(num_modules)

        # Process modules
        if num_lattice_procs <= 1:
            # Process modules in serial
            overlaid_modules = []
            for module, offset_pos, include_mask, _, _ in module_work:
                overlaid = self._overlay_module_worker(module, offset_pos, include_mask, geometry, child_policy)
                overlaid_modules.append(overlaid)
        else:
            # Process modules in parallel
            with ProcessPoolExecutor(max_workers=num_lattice_procs) as executor:
                futures = [
                    executor.submit(self._overlay_module_worker, module, offset_pos, include_mask, geometry, child_policy)
                    for module, offset_pos, include_mask, _, _ in module_work
                ]

                overlaid_modules = [None] * len(module_work)
                for future in as_completed(futures):
                    future_index = futures.index(future)
                    overlaid_modules[future_index] = future.result()

        # Reconstruct the module map with overlaid modules
        new_module_map = [row[:] for row in self.module_map]
        for (_, _, _, i, j), overlaid in zip(module_work, overlaid_modules):
            new_module_map[i][j] = overlaid

        return Lattice(new_module_map)

    @staticmethod
    def _overlay_module_worker(module:         Module,
                               offset:         Tuple[float, float, float],
                               include_mask:   Optional[Module.OverlayMask],
                               geometry:       openmc.Geometry,
                               overlay_policy: PinMesh.OverlayPolicy) -> Module:
        """Worker function for parallel module overlay processing.

        Parameters
        ----------
        module : Module
            The MPACTPy Module to overlay with the OpenMC geometry.
        offset : Tuple[float, float, float]
            The (x, y, z) offset coordinates for the OpenMC geometry relative to
            the module's lower-left corner.
        include_mask : Optional[Module.OverlayMask]
            Optional mask specifying which pins within the module should be
            included in the overlay operation. If None, all pins are included.
        geometry : openmc.Geometry
            The OpenMC Geometry to be overlaid onto the module.
        overlay_policy : PinMesh.OverlayPolicy
            Configuration object specifying overlay method, sampling parameters,
            and process allocation for cascading parallelization.

        Returns
        -------
        Module
            A new Module instance with the OpenMC geometry overlaid.
        """
        return module.overlay(geometry, offset, include_mask, overlay_policy)
