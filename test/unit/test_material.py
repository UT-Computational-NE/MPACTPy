import pytest
from math import isclose
from mpactpy.material import Material
import openmc

def test_material_initialization():
    material = Material(material_type              = 1,
                       density                     = 10.0,
                       temperature                 = 300.0,
                       number_densities            = {"U235": 1e-3, "H": 2e-3},
                       mpact_id                    = 42,
                       thermal_scattering_isotopes = ["H"])

    assert material.mpact_id == 42
    assert material.material_type == 1
    assert isclose(material.density, 10.0)
    assert isclose(material.temperature, 300.0)
    assert material.number_densities == {"U235": 1e-3, "H": 2e-3}
    assert material.thermal_scattering_isotopes == ["H"]

def test_material_equality():
    material1 = Material(material_type=1, density=10.0, temperature=300.0, number_densities={"U235": 1e-3, "O16": 2e-3})
    material2 = Material(material_type=1, density=10.0, temperature=300.0, number_densities={"U235": 1e-3, "O16": 2e-3})
    material3 = Material(material_type=2, density=10.0, temperature=300.0, number_densities={"U235": 1e-3, "O16": 2e-3})

    assert material1 == material2
    assert material1 != material3

def test_material_hash():
    material1 = Material(material_type    = 1,
                         density          = 10.0,
                         temperature      = 300.0,
                         number_densities = {"U235": 1e-3, "O16": 2e-3},
                         mpact_id         = 42)

    # Create a second material with slightly different values within the tolerance
    material2 = Material(material_type    = 1,
                         density          = 10.0 * (1 + 1E-6),
                         temperature      = 300.0 * (1 - 1E-6),
                         number_densities = {"U235": 1e-3 * (1 + 1E-6),
                                             "O16" : 2e-3 * (1 - 1E-6)},
                         mpact_id         = 42)

    assert material1 == material2, "Materials should be equal within tolerance."
    assert hash(material1) == hash(material2), "Hashes should match for equal materials."

def test_material_from_openmc_material():
    openmc_material = openmc.Material()
    openmc_material.set_density('g/cc', 10.0)
    openmc_material.temperature = 300.0
    openmc_material.add_nuclide('U235', 1e-3)
    openmc_material.add_nuclide('H1', 2e-3)

    material = Material.from_openmc_material(material                    = openmc_material,
                                             material_type               = 1,
                                             mpact_id                    = 42,
                                             thermal_scattering_isotopes = ["H1"])

    assert material.mpact_id == 42
    assert material.material_type == 1
    assert isclose(material.density, 10.0)
    assert isclose(material.temperature, 300.0)
    assert material.number_densities["U235"] == openmc_material.get_nuclide_atom_densities()["U235"]
    assert material.number_densities["H1"] == openmc_material.get_nuclide_atom_densities()["H1"]
    assert material.thermal_scattering_isotopes == ["H1"]

def test_isotope_MPACT_ID():
    assert Material.isotope_MPACT_ID("U235", False) == 92235
    assert Material.isotope_MPACT_ID("H1", False) == 1001
    assert Material.isotope_MPACT_ID("C", True) == 6001

def test_material_write_to_string():
    material = Material(material_type               = 1,
                        density                     = 10.0,
                        temperature                 = 300.0,
                        number_densities            = {"U235": 1e-3, "O16": 2e-3},
                        mpact_id                    = 42)

    output = material.write_to_string(prefix="  ")

    expected_output = ("  mat 42 1 10.0 g/cc 300.0 K \\\n"
                       "    8016 0.002\n"
                       "    92235 0.001\n")

    assert output == expected_output
