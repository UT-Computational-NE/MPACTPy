from __future__ import annotations
from typing import List, Dict, Any

from mpactpy.material import Material
from mpactpy.pinmesh import PinMesh
from mpactpy.pin import Pin
from mpactpy.module import Module
from mpactpy.lattice import Lattice
from mpactpy.assembly import Assembly
from mpactpy.core import Core
from mpactpy.utils import list_to_str



class Model():
    """ An input writer for MPACT

    It should noted that MPACT Model does not allow for repeated geometry
    element definitions.  This means that redudant geometry elements
    (i.e. geometry elements that are identical to some previously defined geometry
    element in all features except ID) are purged.

    Attributes
    ----------
    materials : List[Material]
        The model materials
    states : List[Dict[str, str]]
        The model states
    pinmeshes : List[PinMesh]
        The model pinmeshes
    pins : List[Pin]
        The model pins
    modules : List[Module]
        The model modules
    lattices : List[Lattice]
        The model lattices
    assemblies : List[Assembly]
        The model assemblies
    core : Core
        The model core
    mod_dim : Assembly.ModDim
        The x,y,z dimensions of the ray-tracing module
    xsec_settings : Dict[str, str]
        The model cross-section settings
    options : Dict[str, str]
        The model options
    """

    @property
    def materials(self) -> List[Material]:
        return self.core.materials

    @property
    def states(self) -> List[Dict[str, str]]:
        return self._states

    @property
    def pinmeshes(self) -> List[PinMesh]:
        return self.core.pinmeshes

    @property
    def pins(self) -> List[Pin]:
        return self.core.pins

    @property
    def modules(self) -> List[Module]:
        return self.core.modules

    @property
    def lattices(self) -> List[Lattice]:
        return self.core.lattices

    @property
    def assemblies(self) -> List[Assembly]:
        return self.core.assemblies

    @property
    def core(self) -> Core:
        return self._core

    @property
    def mod_dim(self) -> Assembly.ModDim:
        return self.core.mod_dim

    @property
    def xsec_settings(self) -> Dict[str, str]:
        return self._xsec_settings

    @property
    def options(self) -> Dict[str, str]:
        return self._options


    def __init__(self,
                 core          : Core,
                 states        : List[Dict[str, str]],
                 xsec_settings : Dict[str, str],
                 options       : Dict[str, str]
    ):
        self._core          = core
        self._states        = states
        self._xsec_settings = xsec_settings
        self._options       = options

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Model)                  and
                self.core          == other.core          and
                self.states        == other.states        and
                self.xsec_settings == other.xsec_settings and
                self.options       == other.options
               )

    def __hash__(self) -> int:
        return hash((self.core,
                    tuple(tuple(state.items()) for state in self.states),
                    tuple(self.xsec_settings.items()),
                    tuple(self.options.items())))

    def write_to_string(self, caseid: str, indent: int) -> str:
        """ Writes an MPACT input file of the model

        Parameters
        ----------
        caseid : str
            The CASEID of this input
        indent : int
            The length of line indentations

        Returns
        -------
        str
            The MPACT input file as a string
        """

        assert indent > 0

        prefix = "".ljust(indent)

        string = f"CASEID {caseid}\n\n"

        string += "MATERIAL\n"
        for material in self.materials:
            string += material.write_to_string(prefix)
        string += "\n"

        for state in self.states:
            string += "STATE"
            for param, val in state.items():
                string += " " + param + " " + val
            string += "\n"
        string += "\n"

        string += "GEOM\n"
        string += prefix + f"mod_dim {self.mod_dim['X']} {self.mod_dim['Y']} {list_to_str(self.mod_dim['Z'])}\n\n"

        for pinmesh in self.pinmeshes:
            string += pinmesh.write_to_string(prefix)
        string += "\n"

        for pin in self.pins:
            string += pin.write_to_string(prefix)
        string += "\n"

        for module in self.modules:
            string += module.write_to_string(prefix)
        string += "\n"

        for lattice in self.lattices:
            string += lattice.write_to_string(prefix)
        string += "\n"

        for assembly in self.assemblies:
            string += assembly.write_to_string(prefix)
        string += "\n"

        string += self.core.write_to_string(prefix) + "\n"

        string += "XSEC\n"
        for setting, argument in sorted(self.xsec_settings.items()):
            string += prefix + f"{setting} {argument}\n"
        string += "\n"

        string += "OPTION\n"
        for setting, argument in sorted(self.options.items()):
            string += prefix + f"{setting} {argument}\n"
        string += "\n"

        return string

    @staticmethod
    def read_from_string(string_to_read: str) -> Model:
        """ Function for reading an MPACT input file from a string into a model

        Parameters
        ----------
        string_to_read : str
            string representing the mpact input file to be read

        Returns
        -------
            The MPACTModel created from the input file string
        """

        raise NotImplementedError("This method is not yet implemented.")
