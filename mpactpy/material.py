from __future__ import annotations
from typing import Dict, List, Optional, Any
from math import isclose
import openmc

from mpactpy.utils import relative_round, ROUNDING_RELATIVE_TOLERANCE as TOL


class Material():
    """ Class for specifying materials of an MPACT model

    Parameters
        ----------
        material_type : int
            The MPACT material type number
        density : float
            The density (g/cc)
        temperature : float
            The temperature (K)
        number_densities : Dict[str, float]
            The isotopic number densities (atoms/b-cm)
            (key: isotope ID, value: number density)
        mpact_id : int
            The MPACT ID for the material
        thermal_scattering_isotopes : Optional[List[str]]
            List of isotopes that should use thermal scattering libraries

    Attributes
        ----------
        mpact_id : int
            The MPACT ID for the material
        material_type : int
            The MPACT material type number
        density : float
            The density (g/cc)
        temperature : float
            The temperature (K)
        number_densities : Dict[str, float]
            The isotopic number densities (atoms/b-cm)
            (key: isotope ID, value: number density)
        thermal_scattering_isotopes : List[str]
            List of isotopes that should use thermal scattering libraries
    """

    @property
    def mpact_id(self) -> int:
        return self._mpact_id

    @mpact_id.setter
    def mpact_id(self, mpact_id: int) -> None:
        assert mpact_id > 0
        self._mpact_id = mpact_id

    @property
    def material_type(self) -> int:
        return self._material_type

    @property
    def density(self) -> float:
        return self._density

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def number_densities(self) -> Dict[str, float]:
        return self._number_densities

    @property
    def thermal_scattering_isotopes(self) -> List[str]:
        return self._thermal_scattering_isotopes

    def __init__(self,
                 material_type:               int,
                 density:                     float,
                 temperature:                 float,
                 number_densities:            Dict[str, float],
                 mpact_id:                    int = 1,
                 thermal_scattering_isotopes: Optional[List[str]] = None,
    ):

        thermal_scattering_isotopes = [] if thermal_scattering_isotopes is None else thermal_scattering_isotopes

        assert material_type in VALID_MATERIAL_TYPES
        assert density >= 0.
        assert temperature >= 0.
        assert all(number_dens >= 0. for number_dens in number_densities.values())
        assert all(iso in number_densities for iso in thermal_scattering_isotopes)

        self.mpact_id                     = mpact_id
        self._material_type               = material_type
        self._density                     = density
        self._temperature                 = temperature
        self._number_densities            = number_densities
        self._thermal_scattering_isotopes = thermal_scattering_isotopes


    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, Material)                                           and
                self.material_type == other.material_type                             and
                isclose(self.density, other.density, rel_tol=TOL)                     and
                isclose(self.temperature, other.temperature, rel_tol=TOL)             and
                self.number_densities.keys() == other.number_densities.keys()         and
                self.thermal_scattering_isotopes == other.thermal_scattering_isotopes and
                all(isclose(self.number_densities[iso], other.number_densities[iso], rel_tol=TOL)
                    for iso in self.number_densities.keys())
                )

    def __hash__(self) -> int:
        number_densities = sorted({iso: relative_round(numd, TOL)
                                   for iso, numd in self.number_densities.items()})
        return hash((self.material_type,
                     relative_round(self.density, TOL),
                     relative_round(self.temperature, TOL),
                     tuple(number_densities),
                     tuple(self.thermal_scattering_isotopes)))


    @staticmethod
    def from_openmc_material(material:                    openmc.Material,
                             material_type:               int,
                             mpact_id:                    int = 1,
                             thermal_scattering_isotopes: List[str] = []) -> Material:
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
        mpact_id : int
            The MPACT ID for the material
        material : openmc.Material
            The openmc Material with which to build this new material from
        material_type : int
            The MPACT material type of the new MPACT material
        thermal_scattering_isotopes : List[str]
            List of isotopes that should use thermal scattering libraries

        Returns
        -------
        Material
            The MPACT Model material created from the OpenMC Material
        """

        assert material_type in VALID_MATERIAL_TYPES
        assert material.density_units in ['g/cc', 'g/cm3']

        number_densities = {}
        for iso in thermal_scattering_isotopes:
            number_densities[iso] = 0.
        for element in MPACT_NATURAL_ELEMENTS:
            number_densities[element] = 0.
        for iso, number_density in material.get_nuclide_atom_densities().items():
            element = ''.join(filter(str.isalpha, iso))
            if iso in number_densities:
                number_densities[iso] += number_density
            elif element in number_densities:
                number_densities[element] += number_density
            else:
                number_densities[iso] = number_density

        number_densities = {iso: num_dens for iso, num_dens in number_densities.items() if not isclose(num_dens, 0.0)}

        mpact_material = Material(mpact_id                    = mpact_id,
                                  material_type               = material_type,
                                  density                     = material.density,
                                  temperature                 = material.temperature,
                                  number_densities            = number_densities,
                                  thermal_scattering_isotopes = thermal_scattering_isotopes)
        return mpact_material

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

    def write_to_string(self, prefix: str = "") -> str:
        """ Method for writing the material to a string

        It should be noted that this method will only write out those elements / isotopes
        which are currently supported by MPACT (see: MPACT_SUPPORTED_ISOTOPE_IDS defined below).
        Those not supported by MPACT will not be written to the output string.

        Parameters
        ----------
        prefix : str
            A prefix with which to start each line of the written output string

        Returns
        -------
        str
            The string representing the material definition
        """

        string = prefix + f"mat {self.mpact_id} {self.material_type} {self.density} g/cc {self.temperature} K \\\n"

        for iso, number_density in sorted(self.number_densities.items()):
            is_thermal_scattering = iso in self.thermal_scattering_isotopes
            iso = Material.isotope_MPACT_ID(iso, is_thermal_scattering)
            if iso in MPACT_SUPPORTED_ISOTOPE_IDS:
                string += prefix + prefix + f"{iso} {number_density}\n"

        return string


#                             Is_Fluid  Is_Depletable  Has_Resonance_Data  Is_Fuel
VALID_MATERIAL_TYPES = [0, #     F            F                F              F
                        1, #     T            F                F              F
                        2, #     F            T                T              T
                        3, #     F            T                T              F
                        4, #     F            F                T              F
                        5, #     F            T                F              F
                        6, #     T            F                T              F
                        7, #     T            T                F              F
                        8, #     T            T                T              F
                        9] #     T            T                T              T


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
