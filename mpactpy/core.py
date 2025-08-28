from __future__ import annotations
from typing import List, Any, Literal, Dict, Tuple, Optional, TypedDict
from math import isclose
from itertools import accumulate
from concurrent.futures import ProcessPoolExecutor

import openmc

from mpactpy.material import Material
from mpactpy.pinmesh import PinMesh
from mpactpy.pin import Pin
from mpactpy.module import Module
from mpactpy.lattice import Lattice
from mpactpy.assembly import Assembly
from mpactpy.utils import list_to_str, is_rectangular, unique


class Core():
    """ Core of an MPACT model

    Parameters
    ----------
    assembly_map : List[List[Assembly]]
        2-D map of the core assemblies
    symmetry_opt : SymmetryOption
        Core symmetry ("360", "90")
    quarter_sym_opt : QuarterSymmetryOption
        Quarter core centerline symmetry
        (i.e. whether the centerline bisects an assembly through the
        center, or passes between assemblies along the edge)
    min_thickness: float
        The minimum allowed thickness for unionization.  If the unionized mesh
        produces a mesh element with an axial height less than the minimum thickness,
        an error will be thrown.  This is meant as a failsafe for applications in which
        a minimum axial thickness is required.

    Attributes
    ----------
    symmetry_opt : SymmetryOption
        Core symmetry ("360", "90")
    quarter_sym_opt : QuarterSymmetryOption
        Quarter core centerline symmetry
        (i.e. whether the centerline bisects an assembly through the
        center, or passes between assemblies along the edge)
    nx : int
        Number of assemblies along the x-dimension
    ny : int
        Number of assemblies along the y-dimension
    nz : int
        Number of modules along the z-dimension
    height : float
        The total height of the core (cm)
    width : Width
        The total width of the core (cm) in either X or Y directions
        keys: ['X', 'Y']
    pitch : Pitch
        The pitch of each row / column of the core
        keys: ['row', 'column']
    mod_dim : Assembly.ModDim
        The x,y,z dimensions of the ray-tracing module
    assembly_map : List[List[Assembly]]
        2-D map of the core assemblies
    assemblies : List[Assembly]
        The unique assemblies of this core
    lattices : List[Lattice]
        The unique lattices of this core
    modules : List[Module]
        The unique modules of this core
    pins : List[Pin]
        The unique pins of this core
    pinmeshes : List[PinMesh]
        The unique pin meshes of this core
    materials : List[Material]
        The unqiue materials of this core
    """

    SymmetryOption = Literal["360", "90", ""]
    QuarterSymmetryOption = Literal["EDGE", "CENT", ""]

    class Width(TypedDict):
        """ A Typed Dictionary class for Core Radial Widths
        """
        X: float
        Y: float

    class Pitch(TypedDict):
        """ A Typed Dictionary class for Core row / column pitches
        """
        row:    List[float]
        column: List[float]

    @property
    def symmetry_opt(self) -> SymmetryOption:
        return self._symmetry_opt

    @property
    def quarter_sym_opt(self) -> QuarterSymmetryOption:
        return self._quarter_sym_opt

    @property
    def nx(self) -> int:
        return len(self.assembly_map)

    @property
    def ny(self) -> int:
        return len(self.assembly_map[0])

    @property
    def nz(self) -> int:
        return self.assemblies[0].nz

    @property
    def height(self) -> float:
        return self.assemblies[0].height

    @property
    def width(self) -> Width:
        return self._width

    @property
    def pitch(self) -> Pitch:
        return self._pitch

    @property
    def mod_dim(self) -> Assembly.ModDim:
        return self.assemblies[0].mod_dim

    @property
    def assembly_map(self) -> List[List[Assembly]]:
        return self._assembly_map

    @property
    def assemblies(self) -> List[Assembly]:
        return self._assemblies

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


    def __init__(self,
                 assembly_map: List[List[Assembly]],
                 symmetry_opt: SymmetryOption = "",
                 quarter_sym_opt: QuarterSymmetryOption = "",
                 min_thickness: float = 0.):

        assert is_rectangular(assembly_map)

        self._symmetry_opt    = symmetry_opt
        self._quarter_sym_opt = quarter_sym_opt
        self._assembly_map    = assembly_map

        self._assemblies = unique(assembly for row in self.assembly_map for assembly in row if assembly)

        assert len(self.assemblies) > 0

        assert all(isclose(assembly.mod_dim['X'], self.mod_dim['X']) for assembly in self.assemblies)
        assert all(isclose(assembly.mod_dim['Y'], self.mod_dim['Y']) for assembly in self.assemblies)
        assert self._assembly_map_is_radially_internally_continuous()

        assert all(isclose(assembly.height, self.assemblies[0].height) for assembly in self.assemblies if assembly)
        if not self._assemblies_have_same_axial_meshing():
            self._unionize_axial_mesh(min_thickness)

        self._pitch = {'row':    [next((assembly.pitch['Y'] for assembly in row if assembly), 0.0)
                                       for row in self.assembly_map],
                       'column': [next((self.assembly_map[i][j].pitch['X']
                                        for i in range(self.ny) if self.assembly_map[i][j]), 0.0)
                                        for j in range(self.nx)]}

        self._width = {'X': sum(self.pitch["column"]), 'Y': sum(self.pitch["row"])}

        self._lattices   = unique(lattice for assembly in self.assemblies for lattice in assembly.lattices)
        self._modules    = unique(module for lattice in self.lattices for module in lattice.modules)
        self._pins       = unique(pin for module in self.modules for pin in module.pins)
        self._pinmeshes  = unique(pin.pinmesh for pin in self.pins)
        self._materials  = unique(material for pin in self.pins for material in pin.unique_materials)

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Core)                       and
                self.symmetry_opt    == other.symmetry_opt    and
                self.quarter_sym_opt == other.quarter_sym_opt and
                self.assembly_map    == other.assembly_map
               )

    def __hash__(self) -> int:
        return hash((self.symmetry_opt,
                    self.quarter_sym_opt,
                    tuple(tuple(row) for row in self.assembly_map)))

    def write_to_string(self,
                        prefix: str = "",
                        assembly_mpact_ids = Dict[Assembly, int]) -> str:
        """ Method for writing a to a string

        Parameter
        ---------
        prefix : str
            A prefix with which to start each line of the written output string
        assembly_mpact_ids : Dict[Lattice, int]
            A collection of Assemblies and their corresponding MPACT IDs

        Returns
        -------
        str
            The string that represents the
        """

        if assembly_mpact_ids is None:
            assemblies = unique(assembly for row in self.assembly_map for assembly in row if assembly)
            assembly_mpact_ids = {assembly: i+1 for i, assembly in enumerate(assemblies)}

        id_length = max(len(str(assembly_mpact_ids[assembly])) if assembly is not None else 0
                        for row in self.assembly_map for assembly in row)

        string = prefix + "core"
        if len(self.symmetry_opt)    > 0:
            string += f" {self.symmetry_opt}"
        if len(self.quarter_sym_opt) > 0:
            string += f" {self.quarter_sym_opt}"
        string += "\n"
        for row in self.assembly_map:
            assemblies = [assembly_mpact_ids[assembly] if assembly is not None else "" for assembly in row]
            string += prefix + f"  {list_to_str(assemblies, id_length).rstrip()}\n"
        return string

    def _assemblies_have_same_axial_meshing(self)->bool:
        """ A helper method for checking if all assemblies have the same axial axial meshing along their lengths

        Returns
        -------
        True if all assemblies have the same axial meshing, False otherwise
        """

        if any(assembly.nz != self.assemblies[0].nz for assembly in self.assemblies):
            return False

        for assembly in self.assemblies:
            for i in range(assembly.nz):
                if not isclose(assembly.lattice_map[i].pitch['Z'], self.assemblies[0].lattice_map[i].pitch['Z']):
                    return False
        return True

    def _assembly_map_is_radially_internally_continuous(self) -> bool:
        """ A helper method for checking the core assembly map is internally continuous

        This method allows for "stair-case" core boundaries, but does not allow for missing
        assemblies within the core boundaries, or inconsistent assembly pitches along each
        row and column

        Returns
        -------
        True if the assembly map is radially internally continuous, False otherwise
        """

        def is_continuous_line(line, axis):
            prev_assemblies = []
            pitch = None

            for assembly in line:
                if assembly is not None:

                    # Check for inconsistent pitch
                    current_pitch = assembly.pitch[axis]
                    if pitch is not None and not isclose(current_pitch, pitch):
                        return False
                    pitch = current_pitch

                    # Check for holes in the line
                    if len(prev_assemblies) > 2 and prev_assemblies[-1] is None and prev_assemblies[-2] is not None:
                        return False

                prev_assemblies.append(assembly)
            return True

        # Check rows
        for row in self.assembly_map:
            if not is_continuous_line(row, 'Y'):
                return False

        # Check columns
        for j in range(self.ny):
            column = [self.assembly_map[i][j] for i in range(self.nx)]
            if not is_continuous_line(column, 'X'):
                return False

        return True

    def _unionize_axial_mesh(self, min_thickness: float = 0.) -> None:
        """ Helper method that unionizes the axial meshes of the core assemblies

        This method currently only supports cases where all pinmeshes have a single axial material division

        Parameters
        ----------
        min_thickness: float
            The minimum allowed thickness for unionization.  If the unionized mesh
            produces a mesh element with an axial height less than the minimum thickness,
            an error will be thrown.  This is meant as a failsafe for applications in which
            a minimum axial thickness is required.
        """

        meshes = [list(accumulate(lattice.pitch["Z"] for lattice in assembly.lattice_map)) for assembly in self.assemblies]

        all_points = [0.0] + sorted([x for mesh in meshes for x in mesh])
        unionized_mesh = [all_points[0]]
        for point in all_points[1:]:
            if not isclose(point, unionized_mesh[-1]):
                unionized_mesh.append(point)

        for i in range(1, len(unionized_mesh)):
            if unionized_mesh[i] - unionized_mesh[i-1] < min_thickness:
                raise RuntimeError("Axial Mesh Unionization results in a mesh element " + \
                                   "less than the minimum thickness limit: " +\
                                   f"{min_thickness} starting at axial position: {unionized_mesh[i-1]}")

        unionized_assemblies = []
        for assembly in self.assemblies:
            unionized_assemblies.append(None)
            if not assembly is None:
                lattice_map = []
                for start_pos, stop_pos in zip(unionized_mesh[:-1], unionized_mesh[1:]):
                    lattice_map.extend(assembly.get_axial_slice(start_pos, stop_pos).lattice_map)
                unionized_assemblies[-1] = Assembly(lattice_map)

        # This effectively creates a mapping between unionized assemblies and their original unique assembly counterparts
        orig_to_unionized = dict(zip(self.assemblies, unionized_assemblies))

        self._assembly_map = [[orig_to_unionized[assembly] if not(assembly is None) else None
                               for assembly in row] for row in self._assembly_map]

        self._assemblies = unique(unionized_assemblies)

        assert self._assemblies_have_same_axial_meshing()


    def get_axial_slice(self, start_pos: float, stop_pos: float) -> Core:
        """ Method for creating a new Core from an axial slice of this Core

        Parameters
        ----------
        start_pos : float
            The starting axial position of the slice
        stop_pos : float
            The stopping axial position of the slice

        Returns
        -------
        Core
            The new Core created from the axial slice
        """

        assert stop_pos > start_pos

        if stop_pos  <= 0.          or isclose(stop_pos,  0.) or \
           start_pos >= self.height or isclose(start_pos, self.height):
            return None

        start_pos = max(0,           start_pos)
        stop_pos  = min(self.height, stop_pos)

        assembly_map = [[assembly.get_axial_slice(start_pos, stop_pos) if assembly else None
                         for assembly in row] for row in self.assembly_map]

        return Core(assembly_map, self.symmetry_opt, self.quarter_sym_opt)


    def with_height(self, height: float) -> Core:
        """ Method for getting changing core height of 2D cores

        Parameters
        ----------
        height : float
            The height of the new 2D Core

        Returns
        -------
        Core
            The new core with the new height
        """

        assert self.nz == 1, f"nz = {self.nz}, Core must be strictly 2D"

        return Core([[assembly.with_height(height) if assembly else None
                      for assembly in row] for row in self.assembly_map])


    OverlayMask = Dict[Assembly, Optional[Assembly.OverlayMask]]

    def overlay(self,
                geometry:       openmc.Geometry,
                offset:         Tuple[float, float, float] = (0.0, 0.0, 0.0),
                include_only:   Optional[OverlayMask] = None,
                overlay_policy: PinMesh.OverlayPolicy = PinMesh.OverlayPolicy()) -> Core:
        """ A method for overlaying an OpenMC geometry over top an MPACTPy Core

        Parameters
        ----------
        geometry : openmc.Geometry
            The OpenMC Geometry to be mapped onto the MPACTPy Core
        offset : Tuple(float, float, float)
            Offset of the OpenMC geometry's lower-left corner relative to the
            MPACT Core lower-left. Default is (0.0, 0.0, 0.0)
        include_only : Optional[OverlayMask]
            Specifies which MPACT elements should be considered during overlay.
            If None, all elements are included.
        overlay_policy : OverlayPolicy
            A configuration object specifying how a mesh overlay should be done.

        Returns
        -------
        Core
            A new MPACTPy Core which is a copy of the original,
            but with the OpenMC Geometry overlaid on top.
        """

        include_only: Core.OverlayMask = include_only if include_only else \
                                         {assembly: None for row in self.assembly_map for assembly in row if assembly}

        x0, y0, z0 = offset

        # Collect all assembly work to be done
        assembly_work = []
        y = y0 + self.width['Y']
        for i, row in enumerate(self.assembly_map):
            x = x0
            y -= self.pitch['row'][i]
            for j, assembly in enumerate(row):
                if assembly and assembly in include_only:
                    if assembly.has_overlay_work(include_only[assembly]):
                        assembly_work.append((assembly, (x, y, z0), include_only[assembly], i, j))
                x += self.pitch['column'][j]

        # Determine parallelization strategy
        num_assemblies = len(assembly_work)
        num_core_procs = min(num_assemblies, overlay_policy.num_procs)
        child_policy   = overlay_policy.allocate_processes(num_assemblies)

        # Process assemblies in parallel
        with ProcessPoolExecutor(max_workers=num_core_procs) as executor:
            futures = [
                executor.submit(self._overlay_assembly_worker, assembly, offset_pos, include_mask, geometry, child_policy)
                for assembly, offset_pos, include_mask, _, _ in assembly_work
            ]
            overlaid_assemblies = [future.result() for future in futures]

        # Reconstruct the assembly map with overlaid assemblies
        new_assembly_map = [row[:] for row in self.assembly_map]
        for (_, _, _, i, j), overlaid in zip(assembly_work, overlaid_assemblies):
            new_assembly_map[i][j] = overlaid

        return Core(new_assembly_map)

    @staticmethod
    def _overlay_assembly_worker(assembly:       Assembly,
                                 offset:         Tuple[float, float, float],
                                 include_mask:   Optional[Assembly.OverlayMask],
                                 geometry:       openmc.Geometry,
                                 overlay_policy: PinMesh.OverlayPolicy) -> Assembly:
        """Worker function for parallel assembly overlay processing.

        Parameters
        ----------
        assembly : Assembly
            The MPACTPy Assembly to overlay with the OpenMC geometry.
        offset : Tuple[float, float, float]
            The (x, y, z) offset coordinates for the OpenMC geometry relative to
            the assembly's lower-left corner.
        include_mask : Optional[Assembly.OverlayMask]
            Optional mask specifying which lattices within the assembly should be
            included in the overlay operation. If None, all lattices are included.
        geometry : openmc.Geometry
            The OpenMC Geometry to be overlaid onto the assembly.
        overlay_policy : PinMesh.OverlayPolicy
            Configuration object specifying overlay method, sampling parameters,
            and process allocation for cascading parallelization.

        Returns
        -------
        Assembly
            A new Assembly instance with the OpenMC geometry overlaid.
        """
        return assembly.overlay(geometry, offset, include_mask, overlay_policy)
