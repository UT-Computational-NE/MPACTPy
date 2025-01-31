from __future__ import annotations
from typing import List, Any, Literal, Dict
from math import isclose

from mpactpy.material import Material
from mpactpy.pinmesh import PinMesh
from mpactpy.pin import Pin
from mpactpy.module import Module
from mpactpy.lattice import Lattice
from mpactpy.assembly import Assembly
from mpactpy.utils import list_to_str, is_rectangular, unique


class Core():
    """ Core of an MPACT model

    Attributes
    ----------
    symmetry_opt : SymmetryOption
        Core symmetry ("360", "90")
    quarter_sym_opt : QuarterSymmetryOption
        Quarter core centerline symmetry
        (i.e. whether the centerline bisects an assembly through the
        center, or passes between assemblies along the edge)
    nx : int
        Number of modules along the x-dimension
    ny : int
        Number of modules along the y-dimension
    nz : int
        Number of modules along the z-dimension
    height : float
        The total height of the core (cm)
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
                 quarter_sym_opt: QuarterSymmetryOption = ""):

        assert is_rectangular(assembly_map)

        self._symmetry_opt    = symmetry_opt
        self._quarter_sym_opt = quarter_sym_opt
        self._assembly_map    = assembly_map

        self._assemblies = unique(assembly for row in self.assembly_map for assembly in row if assembly)
        self._lattices   = unique(lattice for assembly in self.assemblies for lattice in assembly.lattice_map)
        self._modules    = unique(module for lattice in self.lattices for row in lattice.module_map for module in row)
        self._pins       = unique(pin for module in self.modules for row in module.pin_map for pin in row)
        self._pinmeshes  = unique(pin.pinmesh for pin in self.pins)
        self._materials  = unique(material for pin in self.pins for material in pin.materials)

        assert len(self.assemblies) > 0

        assert all(isclose(assembly.mod_dim['X'], self.mod_dim['X']) for assembly in self.assemblies)
        assert all(isclose(assembly.mod_dim['Y'], self.mod_dim['Y']) for assembly in self.assemblies)
        assert all(assembly.nz == self.assemblies[0].nz for assembly in self.assemblies)
        assert self._assemblies_have_same_axial_spacing()
        assert self._assembly_map_is_radially_internally_continuous()

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
            string += prefix + f"  {list_to_str(assemblies, id_length)}\n"
        return string

    def _assemblies_have_same_axial_spacing(self)->bool:
        """ A helper method for checking that all assemblies have the same axial spacing along their lengths
        """

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
