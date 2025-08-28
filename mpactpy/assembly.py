from __future__ import annotations
from typing import List, Any, Tuple, Optional, Dict, TypedDict
from math import isclose
from concurrent.futures import ProcessPoolExecutor, as_completed

import openmc

from mpactpy.lattice import PinMesh, Lattice
from mpactpy.module import Module
from mpactpy.pin import Pin
from mpactpy.material import Material
from mpactpy.utils import list_to_str, unique


class Assembly():
    """ Assembly of an MPACT model

    Parameters
    ----------
    lattice_map : List[Lattice]
        1-D array of lattice names

    Attributes
    ----------
    pitch : Pitch
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
        The unique lattices contained in this assembly
    modules : List[Module]
        The unique modules contained in this assembly
    pins : List[Pin]
        The unique pins contained in this assembly
    pinmeshes : List[PinMesh]
        The unique pinmeshes contained in this assembly
    materials : List[Material]
        The unique materials contained in this assembly
    """

    class Pitch(TypedDict):
        """ A Typed Dictionary class for Assembly Pitches
        """
        X: float
        Y: float

    class ModDim(TypedDict):
        """ A Typed Dictionary class for Assembly Module Dimensions
        """
        X: float
        Y: float
        Z: List[float]

    @property
    def pitch(self) -> Pitch:
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

        self._lattices  = unique(self.lattice_map)
        self._modules   = unique([module for lattice in self.lattices for module in lattice.modules])
        self._pins      = unique([pin for module in self.modules for pin in module.pins])
        self._pinmeshes = unique([pin.pinmesh for pin in self.pins])
        self._materials = unique([material for pin in self.pins for material in pin.unique_materials])

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



    OverlayMask = Dict[Lattice, Optional[Lattice.OverlayMask]]

    def has_overlay_work(self, include_only: Optional[OverlayMask] = None) -> bool:
        """Check if this assembly has actual overlay work to do based on the include mask.

        Parameters
        ----------
        include_only : Optional[OverlayMask]
            The dictionary of lattices and their masks to include for this assembly

        Returns
        -------
        bool
            True if assembly has overlay work to do, False otherwise
        """
        if include_only is None:
            # No mask means include all lattices in this assembly
            return True

        # Check if assembly contains any lattices that have overlay work to do
        for lattice in self.lattices:
            if lattice in include_only:
                if lattice.has_overlay_work(include_only[lattice]):
                    return True

        return False

    def overlay(self,
                geometry:       openmc.Geometry,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                include_only:   Optional[OverlayMask] = None,
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy()) -> Assembly:
        """ A method for overlaying an OpenMC geometry over top an MPACTPy Assembly

        Parameters
        ----------
        geometry : openmc.Geometry
            The OpenMC Geometry to be mapped onto the MPACTPy Assembly
        offset : Tuple(float, float, float)
            Offset of the OpenMC geometry's lower-left corner relative to the
            MPACT Assembly lower-left. Default is (0.0, 0.0, 0.0)
        include_only : Optional[OverlayMask]
            Specifies which MPACT elements should be considered during overlay.
            If None, all elements are included.
        overlay_policy : OverlayPolicy
            A configuration object specifying how a mesh overlay should be done.

        Returns
        -------
        Assembly
            A new MPACTPy Assembly which is a copy of the original,
            but with the OpenMC Geometry overlaid on top.
        """

        include_only: Assembly.OverlayMask = include_only if include_only else \
                                            {lattice: None for lattice in self.lattice_map}

        x0, y0, z0 = offset

        # Collect all lattice work to be done
        lattice_work = []
        z = z0
        for i, lattice in enumerate(self.lattice_map):
            if lattice in include_only:
                if lattice.has_overlay_work(include_only[lattice]):
                    lattice_work.append((lattice, (x0, y0, z), include_only[lattice], i))
            z += lattice.pitch['Z']

        # Determine parallelization strategy
        num_lattices       = len(lattice_work)
        num_assembly_procs = min(num_lattices, overlay_policy.num_procs)
        child_policy       = overlay_policy.allocate_processes(num_lattices)

        # Process lattices in parallel
        with ProcessPoolExecutor(max_workers=num_assembly_procs) as executor:
            futures = [
                executor.submit(self._overlay_lattice_worker, lattice, offset_pos, include_mask, geometry, child_policy)
                for lattice, offset_pos, include_mask, _ in lattice_work
            ]

            overlaid_lattices = [None] * len(lattice_work)
            for future in as_completed(futures):
                future_index = futures.index(future)
                overlaid_lattices[future_index] = future.result()

        # Reconstruct the lattice map with overlaid lattices
        new_lattice_map = self.lattice_map[:]
        for (_, _, _, i), overlaid in zip(lattice_work, overlaid_lattices):
            new_lattice_map[i] = overlaid

        return Assembly(new_lattice_map)

    @staticmethod
    def _overlay_lattice_worker(lattice:        Lattice,
                                offset:         Tuple[float, float, float],
                                include_mask:   Optional[Lattice.OverlayMask],
                                geometry:       openmc.Geometry,
                                overlay_policy: PinMesh.OverlayPolicy) -> Lattice:
        """Worker function for parallel lattice overlay processing.

        Parameters
        ----------
        lattice : Lattice
            The MPACTPy Lattice to overlay with the OpenMC geometry.
        offset : Tuple[float, float, float]
            The (x, y, z) offset coordinates for the OpenMC geometry relative to
            the lattice's lower-left corner.
        include_mask : Optional[Lattice.OverlayMask]
            Optional mask specifying which modules within the lattice should be
            included in the overlay operation. If None, all modules are included.
        geometry : openmc.Geometry
            The OpenMC Geometry to be overlaid onto the lattice.
        overlay_policy : PinMesh.OverlayPolicy
            Configuration object specifying overlay method, sampling parameters,
            and process allocation for cascading parallelization.

        Returns
        -------
        Lattice
            A new Lattice instance with the OpenMC geometry overlaid.
        """
        return lattice.overlay(geometry, offset, include_mask, overlay_policy)
