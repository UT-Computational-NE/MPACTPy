import pytest
from math import isclose
from numpy.testing import assert_allclose

from mpactpy.module import Module
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh, pinmesh_2D
from test.unit.test_pin import pin, equal_pin, unequal_pin, pin_2D


@pytest.fixture
def module(pin):
    return Module(1, [[pin, pin],
                      [pin, pin]])

@pytest.fixture
def equal_module(equal_pin):
    return Module(1, [[equal_pin, equal_pin],
                      [equal_pin, equal_pin]])

@pytest.fixture
def unequal_module(unequal_pin):
    return Module(1, [[unequal_pin, unequal_pin],
                      [unequal_pin, unequal_pin]])

@pytest.fixture
def module_2D(pin_2D):
    return Module(1, [[pin_2D, pin_2D],
                      [pin_2D, pin_2D]])

def test_module_initialization(module, pin):
    assert module.nx == 2
    assert module.ny == 2
    assert module.nz == 1
    assert_allclose([module.pitch[i] for i in ['X','Y','Z']], [4., 4., 3.])
    assert all([p == pin for row in module.pin_map for p in row])

def test_module_equality(module, equal_module, unequal_module):
    assert module == equal_module
    assert module != unequal_module

def test_module_hash(module, equal_module, unequal_module):
    assert hash(module) == hash(equal_module)
    assert hash(module) != hash(unequal_module)

def test_module_write_to_string(module, pin):
    output = module.write_to_string(prefix="  ",
                                    pin_mpact_ids={pin: 5},
                                    module_mpact_ids={module: 3})
    expected_output = "  module 3 2 2 1\n" + \
                      "    5 5\n" + \
                      "    5 5\n"
    assert output == expected_output

def test_module_get_axial_slice(module):
    module_slice = module.get_axial_slice(0.5, 1.5)
    pin_slice    = module_slice.pin_map[0][0]

    assert pin_slice.pinmesh.number_of_material_regions == 8
    assert pin_slice.pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6]
    assert_allclose([module_slice.pitch[i] for i in ['X','Y','Z']], [4., 4., 1.])
    assert_allclose(pin_slice.pinmesh.zvals, [0.5, 1.0])

def test_module_with_height(module_2D):
    module = module_2D.with_height(3.0)
    assert isclose(module.pitch['Z'], 3.0)