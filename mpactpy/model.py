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

    The STATE, XSEC, OPTION and TRANS blocks may each be given as a KeyValueBlock or as
    a dictionary of keywords and values.  A KeyValueBlock is written in its own order,
    with repeated keywords and comments.  A dictionary is written as before: STATE
    keywords in dictionary order, XSEC and OPTION keywords sorted.

    Parameters
    ----------
    states : List[Settings]
        The model states, one STATE block each, in the order they are run
    core : Core
        The model core
    xsec_settings : Settings
        The model cross-section settings
    options : Settings
        The model options
    trans : Optional[Settings]
        The model transient settings (no TRANS block is written if None)

    Attributes
    ----------
    states : List[Settings]
        The model states, as given
    core : Core
        The model core
    xsec_settings : Settings
        The model cross-section settings, as given
    options : Settings
        The model options, as given
    trans : Optional[Settings]
        The model transient settings, as given
    """

    @property
    def states(self) -> List[Settings]:
        return self._states

    @property
    def core(self) -> Core:
        return self._core

    @property
    def xsec_settings(self) -> Settings:
        return self._xsec_settings

    @property
    def options(self) -> Settings:
        return self._options

    @property
    def trans(self) -> Optional[Settings]:
        return self._trans


    def __init__(self,
                 core          : Core,
                 states        : List[Settings],
                 xsec_settings : Settings,
                 options       : Settings,
                 trans         : Optional[Settings] = None
    ):
        for state in states:
            self._check_settings(state, "STATE")
        self._check_settings(xsec_settings, "XSEC")
        self._check_settings(options,       "OPTION")
        if trans is not None:
            self._check_settings(trans, "TRANS")

        self._core          = core
        self._states        = states
        self._xsec_settings = xsec_settings
        self._options       = options
        self._trans         = trans

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Model)                      and
                self.core            == other.core            and
                self._state_blocks() == other._state_blocks() and
                self._xsec_block()   == other._xsec_block()   and
                self._option_block() == other._option_block() and
                self._trans_block()  == other._trans_block()
               )

    def __hash__(self) -> int:
        return hash((self.core,
                    tuple(self._state_blocks()),
                    self._xsec_block(),
                    self._option_block(),
                    self._trans_block()))

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

        prefix = "".ljust(indent)

        string = f"CASEID {caseid}\n\n"

        string += "MATERIAL\n"
        for material in self.core.materials:
            string += material.write_to_string(prefix, material_ids)
        string += "\n"

        for state in self._state_blocks():
            string += state.write_to_string(prefix)
        string += "\n"

        trans = self._trans_block()
        if trans is not None:
            string += trans.write_to_string(prefix) + "\n"

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

        string += self._xsec_block().write_to_string(prefix) + "\n"

        string += self._option_block().write_to_string(prefix) + "\n"

        return string

    @staticmethod
    def _check_settings(settings: Settings, name: str) -> None:
        if isinstance(settings, KeyValueBlock):
            assert settings.name.upper() == name, f"expected a {name} block, got {settings.name}"
        else:
            assert isinstance(settings, dict), f"{name} settings must be a KeyValueBlock or a dict: {settings}"

    @staticmethod
    def _as_block(settings: Settings, name: str, sort_keys: bool) -> KeyValueBlock:
        # Dictionaries are converted each time they are used, so later changes to them
        # are still written (as they were before KeyValueBlock existed)
        if isinstance(settings, KeyValueBlock):
            return settings
        items = sorted(settings.items()) if sort_keys else settings.items()
        return KeyValueBlock.from_dict(name, dict(items))

    def _state_blocks(self) -> List[KeyValueBlock]:
        return [self._as_block(state, "STATE", sort_keys=False) for state in self.states]

    def _xsec_block(self) -> KeyValueBlock:
        return self._as_block(self.xsec_settings, "XSEC", sort_keys=True)

    def _option_block(self) -> KeyValueBlock:
        return self._as_block(self.options, "OPTION", sort_keys=True)

    def _trans_block(self) -> Optional[KeyValueBlock]:
        return None if self.trans is None else self._as_block(self.trans, "TRANS", sort_keys=False)

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
