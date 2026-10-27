from __future__ import annotations
from typing import List, Dict, Any, Optional, Union

from mpactpy.core import Core
from mpactpy.key_value_block import KeyValueBlock
from mpactpy.utils import list_to_str, num_to_str


Settings = Union[KeyValueBlock, Dict[str, str]]


class Model():
    """ An input writer for MPACT

    It should noted that MPACT Model does not allow for repeated geometry
    element definitions.  This means that redudant geometry elements
    (i.e. geometry elements that are identical to some previously defined geometry
    element in all features except ID) are purged.

    Settings may be given as a KeyValueBlock or a dictionary, and are stored as
    KeyValueBlocks.  XSEC and OPTION dictionaries are written with their keys sorted.

    Parameters
    ----------
    states : List[Settings]
        The model states
    core : Core
        The model core
    xsec_settings : Settings
        The model cross-section settings
    options : Settings
        The model options
    trans : Optional[Settings]
        The model transient settings (optional)

    Attributes
    ----------
    states : List[KeyValueBlock]
        The model states
    core : Core
        The model core
    xsec_settings : KeyValueBlock
        The model cross-section settings
    options : KeyValueBlock
        The model options
    trans : Optional[KeyValueBlock]
        The model transient settings
    """

    @property
    def states(self) -> List[KeyValueBlock]:
        return self._states

    @property
    def core(self) -> Core:
        return self._core

    @property
    def xsec_settings(self) -> KeyValueBlock:
        return self._xsec_settings

    @property
    def options(self) -> KeyValueBlock:
        return self._options

    @property
    def trans(self) -> Optional[KeyValueBlock]:
        return self._trans


    def __init__(self,
                 core          : Core,
                 states        : List[Settings],
                 xsec_settings : Settings,
                 options       : Settings,
                 trans         : Optional[Settings] = None
    ):
        self._core          = core
        self._states        = [self._as_block(state, "STATE") for state in states]
        self._xsec_settings = self._as_block(xsec_settings, "XSEC",   sort_keys=True)
        self._options       = self._as_block(options,       "OPTION", sort_keys=True)
        self._trans         = None if trans is None else self._as_block(trans, "TRANS")

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Model)                  and
                self.core          == other.core          and
                self.states        == other.states        and
                self.xsec_settings == other.xsec_settings and
                self.options       == other.options       and
                self.trans         == other.trans
               )

    def __hash__(self) -> int:
        return hash((self.core,
                    tuple(self.states),
                    self.xsec_settings,
                    self.options,
                    self.trans))

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

        material_ids = {id(material): i+1 for i, material  in enumerate(self.core.materials)}
        pinmesh_ids  = {pinmesh:      i+1 for i, pinmesh   in enumerate(self.core.pinmeshes)}
        pin_ids      = {pin:          i+1 for i, pin       in enumerate(self.core.pins)}
        module_ids   = {module:       i+1 for i, module    in enumerate(self.core.modules)}
        lattice_ids  = {lattice:      i+1 for i, lattice   in enumerate(self.core.lattices)}
        assembly_ids = {assembly:     i+1 for i, assembly  in enumerate(self.core.assemblies)}

        prefix = "".ljust(indent)

        string = f"CASEID {caseid}\n\n"

        string += "MATERIAL\n"
        for material in self.core.materials:
            string += material.write_to_string(prefix, material_ids)
        string += "\n"

        for state in self.states:
            string += state.write_to_string(prefix)
        string += "\n"

        if self.trans is not None:
            string += self.trans.write_to_string(prefix) + "\n"

        string += "GEOM\n"
        string += prefix + f"mod_dim {num_to_str(self.core.mod_dim['X'])} " + \
                                   f"{num_to_str(self.core.mod_dim['Y'])} " + \
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

        string += self.xsec_settings.write_to_string(prefix) + "\n"

        string += self.options.write_to_string(prefix) + "\n"

        return string

    @staticmethod
    def _as_block(settings: Settings, name: str, sort_keys: bool = False) -> KeyValueBlock:
        if isinstance(settings, KeyValueBlock):
            assert settings.name.upper() == name, f"settings.name = {settings.name}, expected {name}"
            return settings
        items = sorted(settings.items()) if sort_keys else settings.items()
        return KeyValueBlock.from_dict(name, dict(items))

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
