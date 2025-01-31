from __future__ import annotations
from typing import List, Dict, Any

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
    states : List[Dict[str, str]]
        The model states
    core : Core
        The model core
    xsec_settings : Dict[str, str]
        The model cross-section settings
    options : Dict[str, str]
        The model options
    """

    @property
    def states(self) -> List[Dict[str, str]]:
        return self._states

    @property
    def core(self) -> Core:
        return self._core

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

        material_ids = {material: i+1 for i, material  in enumerate(self.core.materials)}
        pinmesh_ids  = {pinmesh:  i+1 for i, pinmesh   in enumerate(self.core.pinmeshes)}
        pin_ids      = {pin:      i+1 for i, pin       in enumerate(self.core.pins)}
        module_ids   = {module:   i+1 for i, module    in enumerate(self.core.modules)}
        lattice_ids  = {lattice:  i+1 for i, lattice   in enumerate(self.core.lattices)}
        assembly_ids = {assembly: i+1 for i, assembly  in enumerate(self.core.assemblies)}

        print(material_ids)

        prefix = "".ljust(indent)

        string = f"CASEID {caseid}\n\n"

        string += "MATERIAL\n"
        for material in self.core.materials:
            string += material.write_to_string(prefix, material_ids)
        string += "\n"

        for state in self.states:
            string += "STATE"
            for param, val in state.items():
                string += " " + param + " " + val
            string += "\n"
        string += "\n"

        string += "GEOM\n"
        string += prefix + f"mod_dim {self.core.mod_dim['X']} " + \
                                   f"{self.core.mod_dim['Y']} " + \
                                   f"{list_to_str(self.core.mod_dim['Z'])}\n\n"

        for pinmesh in self.core.pinmeshes:
            string += pinmesh.write_to_string(prefix, pinmesh_ids)
        string += "\n"

        for pin in self.core.pins:
            string += pin.write_to_string(prefix, material_ids, pinmesh_ids, pin_ids)
        string += "\n"

        for module in self.core.modules:
            string += module.write_to_string(prefix, pin_ids, module_ids)
        string += "\n"

        for lattice in self.core.lattices:
            string += lattice.write_to_string(prefix, module_ids, lattice_ids)
        string += "\n"

        for assembly in self.core.assemblies:
            string += assembly.write_to_string(prefix, lattice_ids, assembly_ids)
        string += "\n"

        string += self.core.write_to_string(prefix, assembly_ids) + "\n"

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
