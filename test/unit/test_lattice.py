import pytest
from numpy.testing import assert_allclose

from mpactpy import Lattice
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh
from test.unit.test_pin import pin, equal_pin, unequal_pin
from test.unit.test_module import module, equal_module, unequal_module


@pytest.fixture
def lattice(module):
    return Lattice([[module, module],
                    [module, module]])

@pytest.fixture
def equal_lattice(equal_module):
    return Lattice([[equal_module, equal_module],
                    [equal_module, equal_module]])

@pytest.fixture
def unequal_lattice(unequal_module):
    return Lattice([[unequal_module, unequal_module],
                    [unequal_module, unequal_module]])

def test_lattice_initialization(lattice, module):
    assert lattice.nx == 2
    assert lattice.ny == 2

    assert_allclose([lattice.pitch[i] for i in ['X','Y','Z']], [8., 8., 3.])
    assert all([m == module for row in lattice.module_map for m in row])

def test_lattice_equality(lattice, equal_lattice, unequal_lattice):
    assert lattice == equal_lattice
    assert lattice != unequal_lattice

def test_lattice_hash(lattice, equal_lattice, unequal_lattice):
    assert hash(lattice) == hash(equal_lattice)
    assert hash(lattice) != hash(unequal_lattice)

def test_lattice_write_to_string(lattice, module):
    output = lattice.write_to_string(prefix="  ",
                                     module_mpact_ids={module: 2},
                                     lattice_mpact_ids={lattice: 4})
    expected_output = "  lattice 4 2 2\n" + \
                      "    2 2\n" + \
                      "    2 2\n"
    assert output == expected_output

def test_lattice_get_axial_slice(lattice):
    lattice_slice = lattice.get_axial_slice(0.5, 1.5)
    module_slice  = lattice_slice.module_map[0][0]
    pin_slice     = module_slice.pin_map[0][0]

    assert pin_slice.pinmesh.number_of_material_regions == 8
    assert pin_slice.pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6]
    assert_allclose([lattice_slice.pitch[i] for i in ['X','Y','Z']], [8., 8., 1.])
    assert_allclose(pin_slice.pinmesh.zvals, [0.5, 1.0])
