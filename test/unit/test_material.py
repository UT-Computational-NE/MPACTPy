import pytest
from math import isclose
from mpactpy import Material
from mpactpy.utils import atomic_mass, AVOGADRO, ROUNDING_RELATIVE_TOLERANCE
import openmc

TOL = ROUNDING_RELATIVE_TOLERANCE * 1E-2

@pytest.fixture
def material():
    temp = 300.0
    numd = {"U235": 1e-3, "C": 2e-3}
    spec = Material.MPACTSpecs(thermal_scattering_isotopes = ["C"],
                               is_fluid                    = True,
                               is_depletable               = False,
                               has_resonance               = False,
                               is_fuel                     = False)
    return Material(temp, numd, spec)

@pytest.fixture
def equal_material():
    temp             = 300.0*(1-TOL)
    numd             = {"U235": 1e-3*(1+TOL), "C": 2e-3*(1-TOL)}
    spec = Material.MPACTSpecs(thermal_scattering_isotopes = ["C"],
                               is_fluid                    = True,
                               is_depletable               = False,
                               has_resonance               = False,
                               is_fuel                     = False)
    return Material(temp, numd, spec)

@pytest.fixture
def unequal_material():
    temp = 600.0
    numd = {"U238": 2e-3, "O16": 1e-3}
    spec = Material.MPACTSpecs(thermal_scattering_isotopes = [],
                               is_fluid                    = False,
                               is_depletable               = True,
                               has_resonance               = True,
                               is_fuel                     = True)
    return Material(temp, numd, spec)


def test_material_initialization(material):
    expected_density = sum(num_dens * 1e24 * atomic_mass(iso) / AVOGADRO
                           for iso, num_dens in material.number_densities.items())
    assert isclose(material.density, expected_density)
    assert isclose(material.temperature, 300.0)
    assert material.number_densities == {"U235": 1e-3, "C": 2e-3}
    assert material.thermal_scattering_isotopes == ["C"]

def test_material_equality(material, equal_material, unequal_material):
    assert material == equal_material
    assert material != unequal_material

def test_material_hash(material, equal_material, unequal_material):
    assert hash(material) == hash(equal_material)
    assert hash(material) != hash(unequal_material)

def test_material_from_openmc_material():
    openmc_material = openmc.Material()
    openmc_material.set_density('sum')
    openmc_material.temperature = 300.0
    openmc_material.add_nuclide('U235', 1e-3)
    openmc_material.add_nuclide('C', 2e-3)

    material = Material.from_openmc_material(material    = openmc_material,
                                             mpact_specs = Material.MPACTSpecs(thermal_scattering_isotopes = ["C"]))

    expected_density = sum(num_dens * 1e24 * atomic_mass(iso) / AVOGADRO
                           for iso, num_dens in material.number_densities.items())
    assert isclose(material.density, expected_density)
    assert isclose(material.temperature, 300.0)
    assert material.number_densities["U235"] == openmc_material.get_nuclide_atom_densities()["U235"]
    assert material.number_densities["C"] == openmc_material.get_nuclide_atom_densities()["C"]
    assert material.thermal_scattering_isotopes == ["C"]

def test_isotope_MPACT_ID():
    assert Material.isotope_MPACT_ID("U235", False) == 92235
    assert Material.isotope_MPACT_ID("H1", False) == 1001
    assert Material.isotope_MPACT_ID("C", True) == 6001

def test_material_write_to_string(material):
    expected_density = sum(num_dens * 1e24 * atomic_mass(iso) / AVOGADRO
                           for iso, num_dens in material.number_densities.items())
    output = material.write_to_string(prefix="  ", mpact_ids={material: "42"})
    expected_output = (f"  mat 42 1 {expected_density} g/cc 300.0 K \\\n"
                        "    6001 0.002\n"
                        "    92235 0.001\n")
    assert output == expected_output

def test_material_mix_materials(material, unequal_material):

    policy  = Material.MixPolicy(percent_type='vo', thermal_scattering_policy="intersection")
    mixture = Material.mix_materials([material, unequal_material], [0.5, 0.5], policy)

    expected_num_dens = {"U235": 0.5 * material.number_densities["U235"],
                         "C":    0.5 * material.number_densities["C"],
                         "U238": 0.5 * unequal_material.number_densities["U238"],
                         "O16":  0.5 * unequal_material.number_densities["O16"]}
    assert expected_num_dens.keys() == mixture.number_densities.keys()
    for iso, expected in expected_num_dens.items():
        assert isclose(mixture.number_densities.get(iso, 0.0), expected, rel_tol=TOL)

    expected_temp = 0.5 * material.temperature + 0.5 * unequal_material.temperature
    assert isclose(mixture.temperature, expected_temp, rel_tol=TOL)

    expected_density = 0.5 * material.density + 0.5 * unequal_material.density
    assert isclose(mixture.density, expected_density, rel_tol=TOL)

    assert mixture.is_fluid is False
    assert mixture.is_depletable is True
    assert mixture.has_resonance is True
    assert mixture.is_fuel is True
    assert mixture.thermal_scattering_isotopes == []

    policy  = Material.MixPolicy(percent_type='wo', thermal_scattering_policy="union", temperature=500.0)
    mixture = Material.mix_materials([material, unequal_material], [0.5, 0.5], policy)

    assert expected_num_dens.keys() == mixture.number_densities.keys()
    assert isclose(mixture.temperature, 500.0, rel_tol=TOL)

    inv_densities    = [w / m.density for w, m in zip([0.5, 0.5], [material, unequal_material])]
    vol_fracs        = [inv_rho / sum(inv_densities) for inv_rho in inv_densities]
    expected_density = sum(vf * m.density for vf, m in zip(vol_fracs, [material, unequal_material]))
    assert isclose(mixture.density, expected_density, rel_tol=TOL)
    assert mixture.thermal_scattering_isotopes == ["C"]