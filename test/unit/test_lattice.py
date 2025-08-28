import pytest
from math import isclose
from numpy.testing import assert_allclose

import openmc

from mpactpy import PinMesh, Pin, Module, Lattice
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh, \
                                   rectangular_pinmesh as overlay_mesh, \
                                   pinmesh_2D, openmc_pin, openmc_fuel_material, openmc_moderator_material
from test.unit.test_pin import pin, equal_pin, unequal_pin, pin_2D, overlay_pin, \
                               template_material, template_pin
from test.unit.test_module import module, equal_module, unequal_module, module_2D, \
                                  template_module, overlay_module, openmc_module


@pytest.fixture
def lattice(module):
    return Lattice([[module, module],
                    [module, module]])

@pytest.fixture
def lattice_2D(module_2D):
    return Lattice([[module_2D, module_2D],
                    [module_2D, module_2D]])

@pytest.fixture
def equal_lattice(equal_module):
    return Lattice([[equal_module, equal_module],
                    [equal_module, equal_module]])

@pytest.fixture
def unequal_lattice(unequal_module):
    return Lattice([[unequal_module, unequal_module],
                    [unequal_module, unequal_module]])

@pytest.fixture
def template_lattice(template_module):
    return Lattice([[template_module, template_module],
                   [template_module, template_module]])

@pytest.fixture
def overlay_lattice(overlay_module):
    return Lattice([[overlay_module, overlay_module],
                    [overlay_module, overlay_module]])

@pytest.fixture
def openmc_lattice(openmc_module):
    module = openmc_module.root_universe

    lattice            = openmc.RectLattice(name='2x2 module lattice')
    lattice.pitch      = (6.0, 6.0)
    lattice.lower_left = (-6.0, -6.0)
    lattice.universes  = [[module, module],
                          [module, module]]

    box           = openmc.model.rectangular_prism(12.0, 12.0, boundary_type='reflective')
    lattice_cell  = openmc.Cell(name='lattice cell', fill=lattice, region=box)

    universe  = openmc.Universe(cells=[lattice_cell])
    geometry  = openmc.Geometry(universe)

    return geometry

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

def test_lattice_with_height(lattice_2D):
    lattice = lattice_2D.with_height(3.0)
    assert isclose(lattice.pitch['Z'], 3.0)

def test_module_overlay(openmc_lattice, template_lattice, template_module, template_pin, template_material, overlay_lattice):
    geometry                          = openmc_lattice
    offset                            = (-6.0, -6.0, 0.0)
    overlay_policy                    = PinMesh.OverlayPolicy(method="centroid")
    pin_mask: Pin.OverlayMask         = {template_material}
    module_mask: Module.OverlayMask   = {template_pin : pin_mask}
    include_only: Lattice.OverlayMask = {template_module : module_mask}

    lattice          = template_lattice.overlay(geometry, offset, include_only, overlay_policy)
    expected_lattice = overlay_lattice
    assert lattice == expected_lattice