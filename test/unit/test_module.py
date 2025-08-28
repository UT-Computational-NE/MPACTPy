import pytest
from math import isclose
from numpy.testing import assert_allclose

import openmc

from mpactpy import PinMesh, Pin, Module
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh, \
                                   rectangular_pinmesh as overlay_mesh, \
                                   pinmesh_2D, openmc_pin, openmc_fuel_material, openmc_moderator_material
from test.unit.test_pin import pin, equal_pin, unequal_pin, pin_2D, overlay_pin, \
                               template_material, template_pin


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

@pytest.fixture
def template_module(template_pin):
    return Module(1, [[template_pin, template_pin],
                      [template_pin, template_pin]])

@pytest.fixture
def overlay_module(overlay_pin):
    return Module(1, [[overlay_pin, overlay_pin],
                      [overlay_pin, overlay_pin]])

@pytest.fixture
def openmc_module(openmc_pin):
    pin   = openmc_pin.root_universe

    lattice            = openmc.RectLattice(name='2x2 pin lattice')
    lattice.pitch      = (3.0, 3.0)
    lattice.lower_left = (-3.0, -3.0)
    lattice.universes  = [[pin, pin],
                          [pin, pin]]

    box           = openmc.model.rectangular_prism(6.0, 6.0, boundary_type='reflective')
    lattice_cell  = openmc.Cell(name='lattice cell', fill=lattice, region=box)

    universe  = openmc.Universe(cells=[lattice_cell])
    geometry  = openmc.Geometry(universe)

    return geometry

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

def test_module_with_height(module_2D, pin):
    new_module = module_2D.with_height(3.0)
    assert isclose(new_module.pitch['Z'], 3.0)

    module = Module(2, [[pin, pin],
                        [pin, pin]])

    with pytest.raises(AssertionError, match=f"nz = {module.nz}, Module must be strictly 2D"):
        new_module = module.with_height(3.0)

def test_module_overlay(openmc_module, template_module, template_pin, template_material, overlay_module):
    geometry                         = openmc_module
    offset                           = (-3.0, -3.0, 0.0)
    overlay_policy                   = PinMesh.OverlayPolicy(method="centroid")
    pin_mask: Pin.OverlayMask        = {template_material}
    include_only: Module.OverlayMask = {template_pin : pin_mask}

    module          = template_module.overlay(geometry, offset, include_only, overlay_policy)
    expected_module = overlay_module
    assert module == expected_module