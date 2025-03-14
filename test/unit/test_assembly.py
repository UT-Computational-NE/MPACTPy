import pytest
from math import isclose
from numpy.testing import assert_allclose

from mpactpy import Assembly
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh, pinmesh_2D
from test.unit.test_pin import pin, equal_pin, unequal_pin, pin_2D
from test.unit.test_module import module, equal_module, unequal_module, module_2D
from test.unit.test_lattice import lattice, equal_lattice, unequal_lattice, lattice_2D


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

@pytest.fixture
def assembly_2D(lattice_2D):
    return Assembly([lattice_2D])

def test_assembly_initialization(assembly, lattice):
    assert isclose(assembly.height, 12.0)
    assert assembly.nz == 4

    assert_allclose([assembly.pitch[i] for i in ['X','Y']], [8., 8.])
    assert_allclose([assembly.mod_dim[i] for i in ['X','Y']], [4., 4.])
    assert_allclose(assembly.mod_dim['Z'], [3.])
    assert all([l == lattice for l in assembly.lattice_map])

def test_assembly_equality(assembly, equal_assembly, unequal_assembly):
    assert assembly == equal_assembly
    assert assembly != unequal_assembly

def test_assembly_hash(assembly, equal_assembly, unequal_assembly):
    assert hash(assembly) == hash(equal_assembly)
    assert hash(assembly) != hash(unequal_assembly)

def test_assembly_write_to_string(assembly, lattice):
    output = assembly.write_to_string(prefix="  ",
                                      lattice_mpact_ids={lattice: 4},
                                      assembly_mpact_ids={assembly: 3})
    expected_output = "  assembly 3\n" + \
                      "    4 4 4 4\n"
    assert output == expected_output

def test_assembly_get_axial_slice(assembly):
    assembly_slice = assembly.get_axial_slice(0.5, 1.5)
    lattice_slice  = assembly_slice.lattice_map[0]
    module_slice  = lattice_slice.module_map[0][0]
    pin_slice     = module_slice.pin_map[0][0]

    assert pin_slice.pinmesh.number_of_material_regions == 8
    assert pin_slice.pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6]
    assert isclose(assembly_slice.height, 1.0)
    assert_allclose([assembly_slice.pitch[i] for i in ['X','Y']], [8., 8.])
    assert_allclose(pin_slice.pinmesh.zvals, [0.5, 1.0])

def test_assembly_with_height(assembly, assembly_2D):
    new_assembly = assembly_2D.with_height(3.0)
    assert isclose(new_assembly.height, 3.0)

    with pytest.raises(AssertionError, match=f"nz = {assembly.nz}, Assembly must be strictly 2D"):
        new_assembly = assembly.with_height(3.0)