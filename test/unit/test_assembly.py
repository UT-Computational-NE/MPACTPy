import pytest
from math import isclose
from numpy.testing import assert_allclose

from mpactpy.assembly import Assembly
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh
from test.unit.test_pin import pin, equal_pin, unequal_pin
from test.unit.test_module import module, equal_module, unequal_module
from test.unit.test_lattice import lattice, equal_lattice, unequal_lattice


@pytest.fixture
def assembly(lattice):
    return Assembly([lattice, lattice,
                     lattice, lattice])

@pytest.fixture
def equal_assembly(equal_lattice):
    return Assembly([equal_lattice, equal_lattice,
                     equal_lattice, equal_lattice])

@pytest.fixture
def unequal_assembly(unequal_lattice):
    return Assembly([unequal_lattice, unequal_lattice,
                     unequal_lattice, unequal_lattice])

def test_assembly_initialization(assembly, lattice):
    assert isclose(assembly.height, 12.0)
    assert assembly.nz == 4

    assert_allclose([assembly.pitch[i] for i in ['X','Y']], [8., 8.])
    assert_allclose([assembly.mod_dim[i] for i in ['X','Y']], [4., 4.])
    assert_allclose(assembly.mod_dim['Z'], [3.])
    assert all([l == lattice for l in assembly.lattice_map])
    assert len(assembly.lattices)  == 1
    assert len(assembly.modules)   == 1
    assert len(assembly.pins)      == 1
    assert len(assembly.pinmeshes) == 1
    assert len(assembly.materials) == 1

def test_assembly_equality(assembly, equal_assembly, unequal_assembly):
    assert assembly == equal_assembly
    assert assembly != unequal_assembly

def test_assembly_hash(assembly, equal_assembly, unequal_assembly):
    assert hash(assembly) == hash(equal_assembly)
    assert hash(assembly) != hash(unequal_assembly)

def test_assembly_write_to_string(assembly):
    output = assembly.write_to_string(prefix="  ")
    expected_output = "  assembly 1\n" + \
                      "    1 1 1 1\n"
    assert output == expected_output

def test_assembly_set_unique_elements(lattice):
    assembly = Assembly([lattice, lattice,
                         lattice, lattice])

    other_lattice = lattice
    assembly.set_unique_elements([other_lattice])
    assert all([l is other_lattice for l in assembly.lattice_map])

def test_assembly_get_axial_slice(assembly):
    assembly_slice = assembly.get_axial_slice(0.5, 1.5)
    pin_slice      = assembly_slice.pins[-1]

    assert pin_slice.pinmesh.number_of_material_regions == 8
    assert pin_slice.pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6]
    assert isclose(assembly_slice.height, 1.0)
    assert_allclose([assembly_slice.pitch[i] for i in ['X','Y']], [8., 8.])
    assert_allclose(pin_slice.pinmesh.zvals, [0.5, 1.0])
