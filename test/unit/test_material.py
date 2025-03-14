import pytest
from math import isclose
from mpactpy import Material
from mpactpy.utils import ROUNDING_RELATIVE_TOLERANCE
import openmc

TOL = ROUNDING_RELATIVE_TOLERANCE * 1E-2

@pytest.fixture
def material():
    dens             = 10.0
    temp             = 300.0
    numd             = {"U235": 1e-3, "H": 2e-3}
    thermal_scat_iso = ["H"]
    is_fluid         = True
    is_depl          = False
    has_res          = False
    is_fuel          = False
    return Material(dens, temp, numd, thermal_scat_iso, is_fluid, is_depl, has_res, is_fuel)

@pytest.fixture
def equal_material():
    dens             = 10.0*(1+TOL)
    temp             = 300.0*(1-TOL)
    numd             = {"U235": 1e-3*(1+TOL), "H": 2e-3*(1-TOL)}
    thermal_scat_iso = ["H"]
    is_fluid         = True
    is_depl          = False
    has_res          = False
    is_fuel          = False
    return Material(dens, temp, numd, thermal_scat_iso, is_fluid, is_depl, has_res, is_fuel)

@pytest.fixture
def unequal_material():
    dens             = 5.0
    temp             = 300.0
    numd             = {"U235": 1e-3, "H": 2e-3}
    thermal_scat_iso = ["H"]
    is_fluid         = True
    is_depl          = False
    has_res          = False
    is_fuel          = False
    return Material(dens, temp, numd, thermal_scat_iso, is_fluid, is_depl, has_res, is_fuel)


def test_material_initialization(material):
    assert isclose(material.density, 10.0)
    assert isclose(material.temperature, 300.0)
    assert material.number_densities == {"U235": 1e-3, "H": 2e-3}
    assert material.thermal_scattering_isotopes == ["H"]

def test_material_equality(material, equal_material, unequal_material):
    assert material == equal_material
    assert material != unequal_material

def test_material_hash(material, equal_material, unequal_material):
    assert hash(material) == hash(equal_material)
    assert hash(material) != hash(unequal_material)

def test_material_from_openmc_material():
    openmc_material = openmc.Material()
    openmc_material.set_density('g/cc', 10.0)
    openmc_material.temperature = 300.0
    openmc_material.add_nuclide('U235', 1e-3)
    openmc_material.add_nuclide('H1', 2e-3)

    material = Material.from_openmc_material(material                    = openmc_material,
                                             thermal_scattering_isotopes = ["H1"])

    assert isclose(material.density, 10.0)
    assert isclose(material.temperature, 300.0)
    assert material.number_densities["U235"] == openmc_material.get_nuclide_atom_densities()["U235"]
    assert material.number_densities["H1"] == openmc_material.get_nuclide_atom_densities()["H1"]
    assert material.thermal_scattering_isotopes == ["H1"]

def test_isotope_MPACT_ID():
    assert Material.isotope_MPACT_ID("U235", False) == 92235
    assert Material.isotope_MPACT_ID("H1", False) == 1001
    assert Material.isotope_MPACT_ID("C", True) == 6001

def test_material_write_to_string(material):
    output = material.write_to_string(prefix="  ", mpact_ids={material: "42"})
    expected_output = ("  mat 42 1 10.0 g/cc 300.0 K \\\n"
                       "    1001 0.002\n"
                       "    92235 0.001\n")
    assert output == expected_output
