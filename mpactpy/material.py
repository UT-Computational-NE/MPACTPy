from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from math import isclose

import numpy as np
import openmc

from mpactpy.utils import atomic_mass, AVOGADRO, ROOM_TEMPERATURE, \
                          relative_round, ROUNDING_RELATIVE_TOLERANCE as TOL

class Material():
    """ Class for specifying materials of an MPACT model

    Parameters
    ----------
    temperature : float
        The temperature (K)
    number_densities : Dict[str, float]
        The isotopic number densities (atoms/b-cm)
        (key: isotope ID, value: number density)
    mpact_specs : MPACTSpecs
        Specifications for how the material should be handled in MPACT

    Attributes
    ----------
    density : float
        The density (g/cc)
    temperature : float
        The temperature (K)
    number_densities : Dict[str, float]
        The isotopic number densities (atoms/b-cm)
        (key: isotope ID, value: number density)
    is_fluid : bool
        Boolean flag indicating whether or not the material is a fluid
    is_depletable : bool
        Boolean flag indicating whether or not the material is depletable
    has_resonance : bool
        Boolean flag indicating whether or not the material has resonance data
    is_fuel : bool
        Boolean flag indicating whether or not the material is fuel
    """

    @dataclass
    class MPACTSpecs():
        """ A dataclass for specifications for how a material should be handled in MPACT

        Attributes
        ----------
        replace_isotopes : Dict[str, str]
            Dictionary of isotopes IDs to replace when writing the MPACT input
            (key: original isotope ID, value: replacement isotope ID)
        is_fluid : bool
            Whether the material is a fluid (Default: False)
        is_depletable : bool
            Whether the material is depletable (Default: False)
        has_resonance : bool
            Whether the material has resonance data (Default: False)
        is_fuel : bool
            Whether the material is fuel (Default: False)
        material_type : int
            MPACT numeric encoding for the material type. Automatically set from other flags.

        References
        ----------
        [1] "MPACT Native Input User's Manual", Version 4.4
            Section 4.5 pg 27
        """

        replace_isotopes: Dict[str, str] = field(default_factory=dict)
        is_fluid:         bool = False
        is_depletable:    bool = False
        has_resonance:    bool = False
        is_fuel:          bool = False
        material_type:    int  = 0

        def __post_init__(self):

            material_type_mapping = {# Is_Fluid  Is_Depletable  Has_Resonance_Data  Is_Fuel
                                    (   False,      False,           False,         False): 0,
                                    (   True,       False,           False,         False): 1,
                                    (   False,      True,            True,          True ): 2,
                                    (   False,      True,            True,          False): 3,
                                    (   False,      False,           True,          False): 4,
                                    (   False,      True,            False,         False): 5,
                                    (   True,       False,           True,          False): 6,
                                    (   True,       True,            False,         False): 7,
                                    (   True,       True,            True,          False): 8,
                                    (   True,       True,            True,          True ): 9}

            key = (self.is_fluid, self.is_depletable, self.has_resonance, self.is_fuel)
            assert key in material_type_mapping, f"Invalid material combination: {key}"
            self.material_type = material_type_mapping[key]
            assert all(value in MPACT_SUPPORTED_ISOTOPE_IDS for value in self.replace_isotopes.values()), \
                f"Invalid replacement isotope ID: {self.replace_isotopes.values()}"

    @property
    def density(self) -> float:
        return self._density

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        assert temperature > 0., f"temperature = {temperature}"
        self._temperature = temperature

    @property
    def number_densities(self) -> Dict[str, float]:
        return self._number_densities

    @property
    def replace_isotopes(self) -> Dict[str, str]:
        return self._mpact_specs.replace_isotopes

    @property
    def is_fluid(self) -> bool:
        return self._mpact_specs.is_fluid

    @property
    def is_depletable(self) -> bool:
        return self._mpact_specs.is_depletable

    @property
    def has_resonance(self) -> bool:
        return self._mpact_specs.has_resonance

    @property
    def is_fuel(self) -> bool:
        return self._mpact_specs.is_fuel

    def __init__(self,
                 temperature:                 float,
                 number_densities:            Dict[str, float],
                 mpact_specs:                 MPACTSpecs = None,
    ):

        self._cached_hash = None

        assert all(number_dens >= 0. for number_dens in number_densities.values()), \
            f"number_densities = {number_densities}"

        self.temperature       = temperature
        self._mpact_specs      = mpact_specs if mpact_specs else Material.MPACTSpecs()
        self._number_densities = number_densities
        self._density          = sum(num_dens * 1e24 * atomic_mass(iso) / AVOGADRO
                                     for iso, num_dens in number_densities.items())

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Material)                                           and
                self._mpact_specs.material_type == other._mpact_specs.material_type   and
                isclose(self.temperature, other.temperature, rel_tol=TOL)             and
                self.number_densities.keys() == other.number_densities.keys()         and
                all(isclose(self.number_densities[iso], other.number_densities[iso], rel_tol=TOL)
                    for iso in self.number_densities.keys())
                )

    def __hash__(self) -> int:
        if self._cached_hash is None:
            number_densities = sorted({iso: relative_round(numd, TOL)
                                       for iso, numd in self.number_densities.items()})
            self._cached_hash = hash((self._mpact_specs.material_type,
                                     relative_round(self.temperature, TOL),
                                     tuple(number_densities)))
        return self._cached_hash

    @dataclass
    class MixPolicy():
        """ Data class for specifying how to perform mixing

        Attributes
        ----------
        percent_type : str
            The type of fractions provided. Must be either:
            - 'vo': volume fractions
            - 'wo': weight fractions
        is_fluid : Optional[bool]
            Whether the mixture is a fluid
        is_depletable : Optional[bool]
            Whether the mixture is depletable
        has_resonance : Optional[bool]
            Whether the mixture has resonance data
        is_fuel : Optional[bool]
            Whether the mixture is fuel
        temperature : Optional[float]
            The temperature of the resulting mixture material (K).
        replace_isotope_policy : str
            Policy for how to handle isotope replacements in the mixture.
            Options:
            - 'none'        : Omit all isotope replacements in the mixture (default).
            - 'union'       : Include any isotope replacements that appears in any material.
            - 'intersection': Include only isotopes replacements common to all materials.
            - 'manual'      : Use the provided `replace_isotopes` list.
        replace_isotopes : Dict[str, str]
            Dictionary of isotopes IDs to replace when writing the MPACT input (only used if policy is 'manual')
        """

        percent_type:           str = 'vo'
        is_fluid:               Optional[bool] = None
        is_depletable:          Optional[bool] = None
        has_resonance:          Optional[bool] = None
        is_fuel:                Optional[bool] = None
        temperature:            Optional[float] = None
        replace_isotope_policy: str = 'none'
        replace_isotopes:       Optional[Dict[str, str]] = None

        def __post_init__(self):
            assert self.percent_type in ('vo', 'wo'), f"Invalid percent_type: {self.percent_type}"
            assert self.replace_isotope_policy in ('none', 'manual', 'union', 'intersection'), \
                f"Invalid replace_isotope_policy: {self.replace_isotope_policy}"
            if self.replace_isotope_policy == 'manual':
                assert self.replace_isotopes is not None, \
                    "Must provide replace_isotopes for 'manual' policy"

    @staticmethod
    def mix_materials(materials: List[Material],
                      fracs:     List[float],
                      policy:    MixPolicy = MixPolicy()) -> Material:
        """ Method for mixing materials together

        Parameters
        ----------
        materials : List[Material]
            The materials to be combined
        fracs : List[float]
            The corresponding list of fractions for each material. Must sum to 1.0.
            - If `policy.percent_type` is 'vo', these are volume fractions.
            - If `policy.percent_type` is 'wo', these are weight fractions and are internally
              converted to volume fractions using material densities.
        policy : MixPolicy
            A configuration object specifying how materials should be mixed.

        Returns
        -------
        Material
            Mixture of the materials

        Notes
        -----
        The following are default behaviors if not overridden by a Material.MixPolicy:
        - `is_fluid` will be set to True only if **all** input materials have `is_fluid=True`.
        - `is_depletable`, `has_resonance`, and `is_fuel` will each be set to True if **any**
           of the input materials have the corresponding flag set.
        - Temperature of the resulting mixture material will be a volume-fraction-weighted average.
        """

        assert len(materials) == len(fracs), f"len(materials) = {len(materials)}, len(fracs) = {len(fracs)}"
        assert isclose(sum(fracs), 1.0, rel_tol=1e-6), f"sum(fracs) = {sum(fracs)}"

        if policy.percent_type == 'wo':
            assert all(m.density > 0.0 for m in materials)
            weights  = [f / m.density for m, f in zip(materials, fracs)]
            weights /= np.sum(weights)
        elif policy.percent_type == 'vo':
            weights = fracs

        def coalesce(value, fallback):
            return fallback if value is None else value

        is_fluid      = coalesce(policy.is_fluid,      all(m.is_fluid for m in materials))
        is_depletable = coalesce(policy.is_depletable, any(m.is_depletable for m in materials))
        has_resonance = coalesce(policy.has_resonance, any(m.has_resonance for m in materials))
        is_fuel       = coalesce(policy.is_fuel,       any(m.is_fuel for m in materials))
        temperature   = coalesce(policy.temperature,   sum(m.temperature * w for m, w in zip(materials, weights)))

        replace_isotopes = {}
        if policy.replace_isotope_policy == 'manual':
            replace_isotopes = policy.replace_isotopes
        elif policy.replace_isotope_policy == 'intersection':
            replace_isotopes = dict(set.intersection(*(set(m.replace_isotopes.items()) for m in materials)))
        elif policy.replace_isotope_policy == 'union':
            for m in materials:
                for original_iso, replacement_iso in m.replace_isotopes.items():
                    if original_iso in replace_isotopes:
                        if replace_isotopes[original_iso] != replacement_iso:
                            raise ValueError(f"Conflicting isotope replacement rules for '{original_iso}': "
                                           f"'{replace_isotopes[original_iso]}' vs '{replacement_iso}'")
                    else:
                        replace_isotopes[original_iso] = replacement_iso

        number_densities = defaultdict(float)
        for m, w in zip(materials, weights):
            for iso, num_dens in m.number_densities.items():
                number_densities[iso] += w * num_dens

        specs = Material.MPACTSpecs(replace_isotopes = replace_isotopes,
                                    is_fluid         = is_fluid,
                                    is_depletable    = is_depletable,
                                    has_resonance    = has_resonance,
                                    is_fuel          = is_fuel)

        mixture = Material(temperature      = temperature,
                           number_densities = number_densities,
                           mpact_specs      = specs)
        return mixture


    @staticmethod
    def from_openmc_material(material: openmc.Material, mpact_specs: Optional[MPACTSpecs] = None) -> Material:
        """ Factory method for building an Material from an openmc.Material

        It should be noted that MPACT is limited when modeling certain elements to only be able
        to represent said elements with cross-section data corresponding to natural isotopic
        abudances (see: MPACT_NATURAL_ELEMENTS definition below).  When encountering isotopes
        of these elements, this method sums their number densities together into a single elemental
        number density and assigns the natural abundance element ID.  This ultimately assumes
        that isotopes of said elements may be accurately represented with cross-section data
        corresponding to the elements with natural isotopic abundances.

        Also, if no temperature is specified in the openmc.Material, ROOM_TEMPERATURE is used by default.

        Parameters
        ----------
        material : openmc.Material
            The openmc Material with which to build this new material from
        mpact_specs : Optional[MPACTSpecs]
            Specifications for how the material should be handled in MPACT.
            If none is provided, `Material.MPACTSpecs()` is used as the default

        Returns
        -------
        Material
            The MPACT Model material created from the OpenMC Material
        """

        mpact_specs = Material.MPACTSpecs() if mpact_specs is None else mpact_specs

        number_densities = {}
        for iso in mpact_specs.replace_isotopes.keys():
            number_densities[iso] = 0.
        for element in MPACT_NATURAL_ELEMENTS:
            number_densities[element] = 0.
        for iso, number_density in material.get_nuclide_atom_densities().items():
            iso = str(iso)
            element = ''.join(filter(str.isalpha, iso))
            if iso in number_densities:
                number_densities[iso] += number_density
            elif element in number_densities:
                number_densities[element] += number_density
            else:
                number_densities[iso] = number_density

        number_densities = {iso: num_dens for iso, num_dens in number_densities.items()
                            if not isclose(num_dens, 0.0, abs_tol=1e-20)}

        temperature = material.temperature if material.temperature is not None else ROOM_TEMPERATURE

        mpact_material = Material(temperature      = temperature,
                                  number_densities = number_densities,
                                  mpact_specs      = mpact_specs)
        return mpact_material

    @staticmethod
    def from_openmc_geometry_point(point:       Tuple[float, float, float],
                                   geometry:    openmc.Geometry,
                                   mpact_specs: Optional[Dict[openmc.Material, Material.MPACTSpecs]]
                                  ) -> Optional[Material]:
        """ Factory method for creating an MPACT material based on the OpenMC material
            found at a given point within an OpenMC geometry.

        Parameters
        ----------
        point : Tuple[float, float, float]
            The spatial coordinates at which to query the OpenMC geometry.
        geometry : openmc.Geometry
            The OpenMC geometry to search for material assignments at the given point.
        mpact_specs : Optional[Dict[openmc.Material, Material.MPACTSpecs]]
            Specifications for how the material should be handled in MPACT.
            If none is provided, `Material.MPACTSpecs()` is used as the default

        Returns
        -------
        Optional[Material]
            An MPACT material corresponding to the OpenMC material at the
            given point. Returns None if no material is found.
        """

        mpact_specs = {mat.id: spec for mat, spec in mpact_specs.items()} if mpact_specs else {}
        try:
            elements = geometry.find(tuple(point))
            mat = None
            for item in reversed(elements):
                if isinstance(item, openmc.Cell) and item.fill_type == 'material':
                    mat = item.fill
                    break
            return Material.from_openmc_material(mat, mpact_specs.get(mat.id, Material.MPACTSpecs())) if mat else None
        except RuntimeError:
            return None


    @staticmethod
    def from_openmc_geometry_element(element:     List[Tuple[Union[int, None], float]],
                                     geometry:    openmc.Geometry,
                                     mpact_specs: Optional[Dict[openmc.Material, Material.MPACTSpecs]],
                                     mix_policy:  Optional[Material.MixPolicy] = None
                                    ) -> Optional[Material]:
        """ Factory method for creating an MPACT material based on an MeshMaterialVolume element from an OpenMC geometry.

        Parameters
        ----------
        element : List[Tuple[Union[int, None], float]]
            The OpenMC MeshMaterialVolume element from which to construct the material
        geometry : openmc.Geometry
            The OpenMC geometry corresponding to the MeshMaterialVolume element.
        mpact_specs : Optional[Dict[openmc.Material, Material.MPACTSpecs]]
            Specifications for how the material should be handled in MPACT.
            If none is provided, `Material.MPACTSpecs()` is used as the default
        mix_policy : Optional[Material.MixPolicy]
            Policy for how to mix materials. Used only when `method='homogenized'`.
            If a mix_policy is not found, then Material.MixPolicy() is used by default.

        Returns
        -------
        Optional[Material]
            An MPACT material corresponding to the volume-weighted homogenized OpenMC materials
            located in the specified OpenMC MeshMaterialVolume element. Returns None if no material
            are in the element.
        """

        mpact_specs = {mat.id: spec for mat, spec in mpact_specs.items()} if mpact_specs else {}
        mix_policy = mix_policy or Material.MixPolicy()
        openmc_materials = geometry.get_all_materials()
        if not element:
            return None

        mats = [openmc_materials[mat_id] for mat_id, _ in element]
        mats = [Material.from_openmc_material(mat, mpact_specs.get(mat.id, Material.MPACTSpecs())) for mat in mats]

        vols         = [vol for _, vol in element]
        total_volume = sum(vols)
        fracs        = [vol / total_volume for vol in vols]
        return Material.mix_materials(mats, fracs, mix_policy)


    @staticmethod
    def isotope_MPACT_ID(iso: str, is_thermal_scattering) -> int:
        """ Function for converting an isotope name to its MPACT ID

        Parameters
        ----------
        iso : str
            The name of the isotope to be converted
        is_thermal_scattering
            Is an isotope that uses special thermal scattering treatment

        Returns
        -------
        int
            The MPACT ID corresponding to isotope
        """

        is_element = iso.isalpha()
        if is_element:
            iso = iso + "0"

        atomic_number = openmc.data.zam(iso)[0]
        mass_number   = openmc.data.zam(iso)[1]
        is_metastable = openmc.data.zam(iso)[2]

        ZZZAAAI = atomic_number*1000+int(is_metastable)*100+mass_number+int(is_thermal_scattering)

        return ZZZAAAI

    def write_to_string(self, prefix: str = "", mpact_ids: Dict[Material, int] = None) -> str:
        """ Method for writing the material to a string

        It should be noted that this method will only write out those elements / isotopes
        which are currently supported by MPACT (see: MPACT_SUPPORTED_ISOTOPE_IDS defined below).
        Those not supported by MPACT will not be written to the output string.

        Isotopes which are replaced with a common isotope (as defined in `self.replace_isotopes`)
        will have their number densities summed together and written as the replacement isotope.

        Parameters
        ----------
        prefix : str
            A prefix with which to start each line of the written output string
        mpact_ids : Dict[Material, int]
            A collection of Materials and their corresponding MPACT IDs

        Returns
        -------
        str
            The string representing the material definition
        """

        number_densities = {}
        for iso, num_dens in self.number_densities.items():
            if iso in self.replace_isotopes:
                replacement_iso = self.replace_isotopes[iso]
                if replacement_iso in number_densities:
                    number_densities[replacement_iso] += num_dens
                else:
                    number_densities[replacement_iso] = num_dens
            else:
                number_densities[iso] = num_dens

        mpact_id = 1 if mpact_ids is None else mpact_ids[self]
        string = prefix + f"mat {mpact_id} {self._mpact_specs.material_type} " + \
                          f"{self.density} g/cc {self.temperature} K \\\n"

        for iso, number_density in sorted(number_densities.items()):
            if iso in MPACT_SUPPORTED_ISOTOPE_IDS:
                string += prefix + prefix + f"{MPACT_SUPPORTED_ISOTOPE_IDS[iso]} {number_density}\n"

        return string


# Elements which are only supported in MPACT with natural isotopic abundances
# Jabaay D., Graham A., "MPACT User’s Manual", ORNL,
# ORNL/SPR-2021/2331, https://doi.org/10.2172/1887706 (2022)
# Section 3.4 pg 8-14
MPACT_NATURAL_ELEMENTS = [ 'C', 'Mg', 'Si', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V']


# All MPACT supported elements / isotopes
# Jabaay D., Graham A., "MPACT User’s Manual", ORNL,
# ORNL/SPR-2021/2331, https://doi.org/10.2172/1887706 (2022)
# Section 3.4 pg 8-14
MPACT_SUPPORTED_ISOTOPE_IDS = {"H1": 1001,
                               "H2": 1002,
                               "H3": 1003,
                               "H1_in_CH2": 1006,
                               "H1_in_ZrH": 1040,
                               "He3": 2003,
                               "He4": 2004,
                               "Li6": 3006,
                               "Li7": 3007,
                               "Be9": 4009,
                               "B": 5000,
                               "B10": 5010,
                               "B11": 5011,
                               "C": 6000,
                               "C_in_Graphite": 6001,
                               "N14": 7014,
                               "N15": 7015,
                               "O16_in_UO2": 8001,
                               "O16": 8016,
                               "F19": 9019,
                               "Na23": 11023,
                               "Mg": 12000,
                               "Al27": 13027,
                               "Si": 14000,
                               "P31": 15031,
                               "S": 16000,
                               "Cl": 17000,
                               "K": 19000,
                               "Ca": 20000,
                               "Ti": 22000,
                               "V": 23000,
                               "Cr": 24000,
                               "Cr50": 24050,
                               "Cr52": 24052,
                               "Cr53": 24053,
                               "Cr54": 24054,
                               "Mn55": 25055,
                               "Fe": 26000,
                               "Fe54": 26054,
                               "Fe56": 26056,
                               "Fe57": 26057,
                               "Fe58": 26058,
                               "Co59": 27059,
                               "Ni": 28000,
                               "Ni58": 28058,
                               "Ni60": 28060,
                               "Ni61": 28061,
                               "Ni62": 28062,
                               "Ni64": 28064,
                               "Cu63": 29063,
                               "Cu65": 29065,
                               "Br81": 35581,
                               "Kr82": 36582,
                               "Kr83": 36583,
                               "Kr84": 36584,
                               "Kr85": 36585,
                               "Kr86": 36586,
                               "Sr89": 38589,
                               "Sr90": 38590,
                               "Y89": 39589,
                               "Y90": 39590,
                               "Y91": 39591,
                               "Zr": 40000,
                               "Zr_in_ZrH2": 40001,
                               "Zr90": 40090,
                               "Zr91": 40091,
                               "Zr92": 40092,
                               "Zr94": 40094,
                               "Zr96": 40096,
                               "Zr91_FP": 40591,
                               "Zr93_FP": 40593,
                               "Zr95_FP": 40595,
                               "Zr96_FP": 40596,
                               "Nb93": 41093,
                               "Nb95": 41595,
                               "Mo": 42000,
                               "Mo95": 42095,
                               "Mo95_FP": 42595,
                               "Mo96": 42596,
                               "Mo97": 42597,
                               "Mo98": 42598,
                               "Mo99": 42599,
                               "Mo100": 42600,
                               "Tc99": 43599,
                               "Ru100": 44600,
                               "Ru101": 44601,
                               "Ru102": 44602,
                               "Ru103": 44603,
                               "Ru104": 44604,
                               "Ru105": 44605,
                               "Ru106": 44606,
                               "Rh_in_homogenized_detector": 45001,
                               "Rh_in_virtual_detector": 45002,
                               "Rh103": 45103,
                               "Rh103_FP": 45603,
                               "Rh105": 45605,
                               "Pd104": 46604,
                               "Pd105": 46605,
                               "Pd106": 46606,
                               "Pd107": 46607,
                               "Pd108": 46608,
                               "Ag107": 47107,
                               "Ag109": 47109,
                               "Ag109_FP": 47609,
                               "Ag111": 47611,
                               "Ag110m": 47710,
                               "Cd": 48000,
                               "Cd110": 48110,
                               "Cd111": 48111,
                               "Cd112": 48112,
                               "Cd113": 48113,
                               "Cd114": 48114,
                               "Cd110_FP": 48610,
                               "Cd111_FP": 48611,
                               "Cd113_FP": 48613,
                               "In": 49000,
                               "In113": 49113,
                               "In115": 49115,
                               "In115_FP": 49615,
                               "Sn": 50000,
                               "Sn112": 50112,
                               "Sn114": 50114,
                               "Sn115": 50115,
                               "Sn116": 50116,
                               "Sn117": 50117,
                               "Sn118": 50118,
                               "Sn119": 50119,
                               "Sn120": 50120,
                               "Sn122": 50122,
                               "Sn124": 50124,
                               "Sn125": 50125,
                               "Sb": 51000,
                               "Sb121": 51121,
                               "Sb123": 51123,
                               "Sb124": 51124,
                               "Sb125": 51125,
                               "Sb121_FP": 51621,
                               "Sb125_FP": 51625,
                               "Sb127": 51627,
                               "Te132": 52632,
                               "Te127m": 52727,
                               "Te129m": 52729,
                               "I127": 53627,
                               "I129": 53629,
                               "I131": 53631,
                               "I135": 53635,
                               "Xe128": 54628,
                               "Xe130": 54630,
                               "Xe131": 54631,
                               "Xe132": 54632,
                               "Xe133": 54633,
                               "Xe134": 54634,
                               "Xe135": 54635,
                               "Xe136": 54636,
                               "Xe135m": 54735,
                               "Cs133": 55633,
                               "Cs134": 55634,
                               "Cs135": 55635,
                               "Cs136": 55636,
                               "Cs137": 55637,
                               "Ba134": 56634,
                               "Ba137": 56637,
                               "Ba140": 56640,
                               "La139": 57639,
                               "La140": 57640,
                               "Ce140": 58640,
                               "Ce141": 58641,
                               "Ce142": 58642,
                               "Ce143": 58643,
                               "Ce144": 58644,
                               "Pr141": 59641,
                               "Pr143": 59643,
                               "Nd142": 60642,
                               "Nd143": 60643,
                               "Nd144": 60644,
                               "Nd145": 60645,
                               "Nd146": 60646,
                               "Nd147": 60647,
                               "Nd148": 60648,
                               "Nd150": 60650,
                               "Pm147": 61647,
                               "Pm148": 61648,
                               "Pm149": 61649,
                               "Pm151": 61651,
                               "Pm148m": 61748,
                               "Sm152": 62152,
                               "Sm153": 62153,
                               "Sm147": 62647,
                               "Sm148": 62648,
                               "Sm149": 62649,
                               "Sm150": 62650,
                               "Sm151": 62651,
                               "Sm152_FP": 62652,
                               "Sm153_FP": 62653,
                               "Sm154": 62654,
                               "Eu151": 63151,
                               "Eu152": 63152,
                               "Eu153": 63153,
                               "Eu154": 63154,
                               "Eu155": 63155,
                               "Eu156": 63156,
                               "Eu157": 63157,
                               "Eu151_FP": 63651,
                               "Eu153_FP": 63653,
                               "Eu154_FP": 63654,
                               "Eu155_FP": 63655,
                               "Eu156_FP": 63656,
                               "Eu157_FP": 63657,
                               "Gd152": 64152,
                               "Gd154": 64154,
                               "Gd155": 64155,
                               "Gd156": 64156,
                               "Gd157": 64157,
                               "Gd158": 64158,
                               "Gd160": 64160,
                               "Gd154_FP": 64654,
                               "Gd155_FP": 64655,
                               "Gd156_FP": 64656,
                               "Gd157_FP": 64657,
                               "Gd158_FP": 64658,
                               "Gd160_FP": 64660,
                               "Tb159": 65159,
                               "Tb160": 65160,
                               "Tb161": 65161,
                               "Tb159_FP": 65659,
                               "Tb160_FP": 65660,
                               "Tb161_FP": 65661,
                               "Dy160": 66160,
                               "Dy161": 66161,
                               "Dy162": 66162,
                               "Dy163": 66163,
                               "Dy164": 66164,
                               "Dy160_FP": 66660,
                               "Dy161_FP": 66661,
                               "Dy162_FP": 66662,
                               "Dy163_FP": 66663,
                               "Dy164_FP": 66664,
                               "Ho165": 67165,
                               "Ho165_FP": 67665,
                               "Er162": 68162,
                               "Er164": 68164,
                               "Er166": 68166,
                               "Er167": 68167,
                               "Er168": 68168,
                               "Er170": 68170,
                               "Lu176": 71176,
                               "Hf174": 72174,
                               "Hf176": 72176,
                               "Hf177": 72177,
                               "Hf178": 72178,
                               "Hf179": 72179,
                               "Hf180": 72180,
                               "Ta181": 73181,
                               "Ta182": 73182,
                               "W": 74000,
                               "W182": 74182,
                               "W183": 74183,
                               "W184": 74184,
                               "W186": 74186,
                               "Ir191": 77191,
                               "Ir193": 77193,
                               "Au197": 79197,
                               "Pb206": 82206,
                               "Pb207": 82207,
                               "Pb208": 82208,
                               "Bi209": 83209,
                               "Th230": 90230,
                               "Th232": 90232,
                               "Pa231": 91231,
                               "Pa232": 91232,
                               "Pa233": 91233,
                               "Pa234": 91234,
                               "U232": 92232,
                               "U233": 92233,
                               "U234": 92234,
                               "U235": 92235,
                               "U236": 92236,
                               "U237": 92237,
                               "U238": 92238,
                               "Np237": 93237,
                               "Np238": 93238,
                               "Np239": 93239,
                               "Pu236": 94236,
                               "Pu238": 94238,
                               "Pu239": 94239,
                               "Pu240": 94240,
                               "Pu241": 94241,
                               "Pu242": 94242,
                               "Am241": 95241,
                               "Am242": 95242,
                               "Am243": 95243,
                               "Am242m": 95342,
                               "Cm242": 96242,
                               "Cm243": 96243,
                               "Cm244": 96244,
                               "Cm245": 96245,
                               "Cm246": 96246,
                               "Cm247": 96247,
                               "Cm248": 96248,
                               "Bk249": 97249,
                               "Cf249": 98249,
                               "Cf250": 98250,
                               "Cf251": 98251,
                               "Cf252": 98252}
