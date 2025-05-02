from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from math import isclose

import numpy as np
import openmc

from mpactpy.utils import atomic_mass, AVOGADRO, \
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
    thermal_scattering_isotopes : List[str]
        List of isotopes that should use thermal scattering libraries
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
        thermal_scattering_isotopes : List[str]
            List of isotopes that should use thermal scattering libraries (Default: [])
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

        thermal_scattering_isotopes: List[str] = field(default_factory=list)
        is_fluid:                    bool = False
        is_depletable:               bool = False
        has_resonance:               bool = False
        is_fuel:                     bool = False
        material_type:               int  = 0

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
    def thermal_scattering_isotopes(self) -> List[str]:
        return self._mpact_specs.thermal_scattering_isotopes

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

        assert all(number_dens >= 0. for number_dens in number_densities.values()), \
            f"number_densities = {number_densities}"
        assert all(iso in number_densities for iso in mpact_specs.thermal_scattering_isotopes), \
            f"thermal_scattering_isotopes = {mpact_specs.thermal_scattering_isotopes}"

        self.temperature       = temperature
        self._number_densities = number_densities
        self._mpact_specs      = mpact_specs if mpact_specs else Material.MPACTSpecs()
        self._density          = sum(num_dens * 1e24 * atomic_mass(iso) / AVOGADRO
                                     for iso, num_dens in number_densities.items())

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Material)                                           and
                self._mpact_specs.material_type == other._mpact_specs.material_type   and
                isclose(self.temperature, other.temperature, rel_tol=TOL)             and
                self.number_densities.keys() == other.number_densities.keys()         and
                self.thermal_scattering_isotopes == other.thermal_scattering_isotopes and
                all(isclose(self.number_densities[iso], other.number_densities[iso], rel_tol=TOL)
                    for iso in self.number_densities.keys())
                )

    def __hash__(self) -> int:
        number_densities = sorted({iso: relative_round(numd, TOL)
                                   for iso, numd in self.number_densities.items()})
        return hash((self._mpact_specs.material_type,
                     relative_round(self.temperature, TOL),
                     tuple(number_densities),
                     tuple(self.thermal_scattering_isotopes)))

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
        thermal_scattering_policy : str
            Policy for how to handle thermal scattering isotopes in the mixture.
            Options:
            - 'none'        : Omit all thermal scattering isotopes in the mixture (default).
            - 'union'       : Include any isotope that appears in any material .
            - 'intersection': Include only isotopes common to all materials.
            - 'manual'      : Use the provided `thermal_scattering_isotopes` list.
        thermal_scattering_isotopes : Optional[List[str]]
            Manual override of thermal scattering isotopes to use (only used if policy is 'manual').
        """

        percent_type:                str = 'vo'
        is_fluid:                    Optional[bool] = None
        is_depletable:               Optional[bool] = None
        has_resonance:               Optional[bool] = None
        is_fuel:                     Optional[bool] = None
        temperature:                 Optional[float] = None
        thermal_scattering_policy:   str = 'none'
        thermal_scattering_isotopes: Optional[List[str]] = None

        def __post_init__(self):
            assert self.percent_type in ('vo', 'wo'), f"Invalid percent_type: {self.percent_type}"
            assert self.thermal_scattering_policy in ('none', 'manual', 'union', 'intersection'), \
                f"Invalid thermal_scattering_policy: {self.thermal_scattering_policy}"
            if self.thermal_scattering_policy == 'manual':
                assert self.thermal_scattering_isotopes is not None, \
                    "Must provide thermal_scattering_isotopes for 'manual' policy"

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

        thermal_scattering_isotopes = []
        if policy.thermal_scattering_policy == 'manual':
            thermal_scattering_isotopes = policy.thermal_scattering_isotopes
        elif policy.thermal_scattering_policy == 'union':
            thermal_scattering_isotopes = sorted(set(iso for m in materials
                                                     for iso in m.thermal_scattering_isotopes))
        elif policy.thermal_scattering_policy == 'intersection':
            thermal_scattering_isotopes = sorted(set.intersection(*[set(m.thermal_scattering_isotopes)
                                                                    for m in materials])) if materials else []

        number_densities = defaultdict(float)
        for m, w in zip(materials, weights):
            for iso, num_dens in m.number_densities.items():
                number_densities[iso] += w * num_dens

        specs = Material.MPACTSpecs(thermal_scattering_isotopes = thermal_scattering_isotopes,
                                    is_fluid                    = is_fluid,
                                    is_depletable               = is_depletable,
                                    has_resonance               = has_resonance,
                                    is_fuel                     = is_fuel)

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
        for iso in mpact_specs.thermal_scattering_isotopes:
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

        number_densities = {iso: num_dens for iso, num_dens in number_densities.items() if not isclose(num_dens, 0.0)}

        mpact_material = Material(temperature      = material.temperature,
                                  number_densities = number_densities,
                                  mpact_specs      = mpact_specs)
        return mpact_material

    @staticmethod
    def from_openmc_model_point(point:       Tuple[float, float, float],
                                model:       openmc.Model,
                                mpact_specs: Optional[Dict[openmc.Material, Material.MPACTSpecs]]
                               ) -> Optional[Material]:
        """ Factory method for creating an MPACT material based on the OpenMC material
            found at a given point within an OpenMC model.

        Parameters
        ----------
        point : Tuple[float, float, float]
            The spatial coordinates at which to query the OpenMC model.
        model : openmc.Model
            The OpenMC model to search for material assignments at the given point.
        mpact_specs : Optional[Dict[openmc.Material, Material.MPACTSpecs]]
            Specifications for how the material should be handled in MPACT.
            If none is provided, `Material.MPACTSpecs()` is used as the default

        Returns
        -------
        Optional[Material]
            An MPACT material corresponding to the OpenMC material at the
            given point. Returns None if no material is found.
        """

        mpact_specs = mpact_specs or {}
        try:
            elements = model.geometry.find(tuple(point))
            mat = None
            for item in reversed(elements):
                if isinstance(item, openmc.Cell) and item.fill_type == 'material':
                    mat = item.fill
                    break
            return Material.from_openmc_material(mat, mpact_specs.get(mat, Material.MPACTSpecs())) if mat else None
        except RuntimeError:
            return None


    @staticmethod
    def from_openmc_model_element(element:     List[Tuple[Union[int, None], float]],
                                  model:       openmc.Model,
                                  mpact_specs: Optional[Dict[openmc.Material, Material.MPACTSpecs]],
                                  mix_policy:  Optional[Material.MixPolicy] = None
                                 ) -> Optional[Material]:
        """ Factory method for creating an MPACT material based on an MeshMaterialVolume element from an OpenMC model.

        Parameters
        ----------
        element : List[Tuple[Union[int, None], float]]
            The OpenMC MeshMaterialVolume element from which to construct the material
        model : openmc.Model
            The OpenMC model corresponding to the MeshMaterialVolume element.
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

        mpact_specs = mpact_specs or {}
        mix_policy = mix_policy or Material.MixPolicy()
        openmc_materials = model.geometry.get_all_materials()
        if not element:
            return None

        mats = [openmc_materials[mat_id] for mat_id, _ in element]
        mats = [Material.from_openmc_material(mat, mpact_specs.get(mat, Material.MPACTSpecs())) for mat in mats]

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

        mpact_id = 1 if mpact_ids is None else mpact_ids[self]
        string = prefix + f"mat {mpact_id} {self._mpact_specs.material_type} " + \
                          f"{self.density} g/cc {self.temperature} K \\\n"

        for iso, number_density in sorted(self.number_densities.items()):
            is_thermal_scattering = iso in self.thermal_scattering_isotopes
            iso = Material.isotope_MPACT_ID(iso, is_thermal_scattering)
            if iso in MPACT_SUPPORTED_ISOTOPE_IDS:
                string += prefix + prefix + f"{iso} {number_density}\n"

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
MPACT_SUPPORTED_ISOTOPE_IDS = [ 1001,  1002,  1003,  1006,  1040,
                                2003,  2004,  3006,  3007,
                                4009,
                                5000,  5010,  5011,
                                6000,  6001,
                                7014,  7015,
                                8001,  8016,
                                9019,
                               11023,
                               12000,
                               13027,
                               14000,
                               15031,
                               16000,
                               17000,
                               19000,
                               20000,
                               22000,
                               23000,
                               24000, 24050, 24052, 24053, 24054,
                               25055,
                               26000, 26054, 26056, 26057, 26058,
                               27059,
                               28000,
                               28058, 28060, 28061, 28062, 28064,
                               29063, 29065,
                               35581,
                               36582, 36583, 36584, 36585, 36586,
                               38589, 38590, 39589, 39590, 39591,
                               40000, 40001, 40090, 40091, 40092, 40094, 40096, 40591, 40593, 40595, 40596,
                               41093, 41595,
                               42000, 42095, 42595, 42596, 42597, 42598, 42599, 42600,
                               43599,
                               44600, 44601, 44602, 44603, 44604, 44605, 44606,
                               45001, 45002, 45103, 45603, 45605,
                               46604, 46605, 46606, 46607, 46608,
                               47107, 47109, 47609, 47611, 47710,
                               48000, 48110, 48111, 48112, 48113, 48114,
                               48610, 48611, 48613,
                               49000, 49113, 49115, 49615,
                               50000, 50112, 50114, 50115, 50116, 50117, 50118, 50119, 50120, 50122, 50124, 50125,
                               51000, 51121, 51123, 51124, 51125, 51621, 51625, 51627,
                               52632, 52727, 52729,
                               53627, 53629, 53631, 53635,
                               54628, 54630, 54631, 54632, 54633, 54634, 54635, 54636, 54735,
                               55633, 55634, 55635, 55636, 55637,
                               56634, 56637, 56640,
                               57639, 57640,
                               58640, 58641, 58642, 58643, 58644,
                               59641, 59643,
                               60642, 60643, 60644, 60645, 60646, 60647, 60648, 60650,
                               61647, 61648, 61649, 61651, 61748,
                               62152, 62153, 62647, 62648, 62649, 62650, 62651, 62652, 62653, 62654,
                               63151, 63152, 63153, 63154, 63155, 63156, 63157, 63651, 63653, 63654, 63655, 63656, 63657,
                               64152, 64154, 64155, 64156, 64157, 64158, 64160, 64654, 64655, 64656, 64657, 64658, 64660,
                               65159, 65160, 65161, 65659, 65660, 65661,
                               66160, 66161, 66162, 66163, 66164, 66660, 66661, 66662, 66663, 66664,
                               67165, 67665,
                               68162, 68164, 68166, 68167, 68168, 68170,
                               71176,
                               72174, 72176, 72177, 72178, 72179, 72180,
                               73181, 73182,
                               74000, 74182, 74183, 74184, 74186,
                               77191, 77193,
                               79197,
                               82206, 82207, 82208,
                               83209,
                               90230, 90232,
                               91231, 91232, 91233, 91234,
                               92232, 92233, 92234, 92235, 92236, 92237, 92238,
                               93237, 93238, 93239,
                               94236, 94238, 94239, 94240, 94241, 94242,
                               95241, 95242, 95243, 95342,
                               96242, 96243, 96244, 96245, 96246, 96247, 96248,
                               97249,
                               98249, 98250, 98251, 98252]
