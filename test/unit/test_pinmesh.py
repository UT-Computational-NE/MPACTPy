import pytest
from math import isclose, pi
import shutil

from numpy.testing import assert_allclose
import openmc
openmc.config['openmc'] = shutil.which("openmc")


from mpactpy.utils import ROUNDING_RELATIVE_TOLERANCE
from mpactpy import Material
from mpactpy import PinMesh, RectangularPinMesh, GeneralCylindricalPinMesh

TOL = ROUNDING_RELATIVE_TOLERANCE * 1E-2

@pytest.fixture
def openmc_fuel_material():
    fuel = openmc.Material(name='UO2 Fuel', temperature = 300.0)
    fuel.add_element('U', 1.0, enrichment=4.0)
    fuel.add_element('O', 2.0)
    fuel.set_density('g/cm3', 10.0)
    return fuel


@pytest.fixture
def openmc_moderator_material():
    moderator = openmc.Material(name='Water', temperature = 300.0)
    moderator.add_element('H', 2.0)
    moderator.add_element('O', 1.0)
    moderator.set_density('g/cm3', 1.0)
    return moderator


@pytest.fixture
def openmc_pin(openmc_fuel_material, openmc_moderator_material):
    fuel      = openmc_fuel_material
    moderator = openmc_moderator_material

    fuel_radius = 0.4
    pin_pitch   = 3.0
    fuel_cyl    = openmc.ZCylinder(r=fuel_radius)
    box         = openmc.model.rectangular_prism(pin_pitch, pin_pitch, boundary_type='reflective')

    fuel_cell      = openmc.Cell(name='fuel', fill=fuel, region=-fuel_cyl)
    moderator_cell = openmc.Cell(name='moderator', fill=moderator, region=+fuel_cyl & box)

    universe  = openmc.Universe(cells=[fuel_cell, moderator_cell])
    geometry  = openmc.Geometry(universe)
    materials = openmc.Materials([fuel, moderator])

    return openmc.Model(geometry=geometry, materials=materials)


@pytest.fixture
def pinmesh_2D():
    xvals = [1.0, 2.0, 3.0]
    yvals = [1.0, 2.0, 3.0]
    zvals = [1.0]
    ndivx = [10, 10, 10]
    ndivy = [10, 10, 10]
    ndivz = [1]
    return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz)

@pytest.fixture
def rectangular_pinmesh():
    xvals = [1.0, 2.0, 3.0]
    yvals = [1.0, 2.0, 3.0]
    zvals = [1.0, 2.0, 3.0]
    ndivx = [10, 10, 10]
    ndivy = [10, 10, 10]
    ndivz = [5, 5, 5]
    return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz)

@pytest.fixture
def equal_rectangular_pinmesh():
    xvals = [1.0*(1+TOL), 2.0*(1-TOL), 3.0*(1+TOL)]
    yvals = [1.0*(1-TOL), 2.0*(1+TOL), 3.0*(1+TOL)]
    zvals = [1.0*(1+TOL), 2.0*(1+TOL), 3.0*(1-TOL)]
    ndivx = [10, 10, 10]
    ndivy = [10, 10, 10]
    ndivz = [5, 5, 5]
    return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz)

@pytest.fixture
def unequal_rectangular_pinmesh():
    xvals = [1.0, 2.0, 5.0]
    yvals = [1.0, 2.0, 3.0]
    zvals = [1.0, 2.0, 3.0]
    ndivx = [10, 10, 10]
    ndivy = [10, 10, 10]
    ndivz = [5, 5, 5]
    return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz)


def materials_are_close(lhs:     Material,
                        rhs:     Material,
                        rel_tol: float = 1E-2) -> bool:
    """ Helper function for making sure materials are close

    With the isotopic comparisons, different versions of openmc, and particularly
    different cross-section libraries, will result in different isotopes being
    associated with 'natural' concentrations.  For this testing, the expected_material
    is defined using 'fewer' natural isotopes, and the test material must at least
    have those isotopes.  This allows the test material to have additional `natural`
    isotopes from using different openmc / xs-library versions and the test still pass.
    """

    return (isclose(lhs.density, rhs.density, rel_tol=rel_tol)                 and
            isclose(lhs.temperature, rhs.temperature, rel_tol=rel_tol)         and
            lhs.thermal_scattering_isotopes == rhs.thermal_scattering_isotopes and
            lhs.is_fluid                    == rhs.is_fluid                    and
            lhs.is_depletable               == rhs.is_depletable               and
            lhs.has_resonance               == rhs.has_resonance               and
            lhs.is_fuel                     == rhs.is_fuel                     and
            all(iso in lhs.number_densities.keys() for iso in rhs.number_densities.keys()) and
            all(isclose(lhs.number_densities[iso], rhs.number_densities[iso], rel_tol=rel_tol)
                for iso in rhs.number_densities.keys()))


def test_rectangular_pinmesh_initialization(rectangular_pinmesh):
    pinmesh = rectangular_pinmesh
    assert pinmesh.number_of_material_regions == 27
    assert pinmesh.regions_inside_bounds == [i for i in range(27)]
    assert_allclose([pinmesh.pitch[i] for i in ['X','Y','Z']], [3., 3., 3.])
    assert_allclose(pinmesh.xvals, [1.0, 2.0, 3.0])
    assert_allclose(pinmesh.yvals, [1.0, 2.0, 3.0])
    assert_allclose(pinmesh.zvals, [1.0, 2.0, 3.0])
    assert_allclose(pinmesh.ndivx, [10, 10, 10])
    assert_allclose(pinmesh.ndivy, [10, 10, 10])
    assert_allclose(pinmesh.ndivz, [5, 5, 5])

def test_rectangular_pinmesh_equality(rectangular_pinmesh,
                                      equal_rectangular_pinmesh,
                                      unequal_rectangular_pinmesh):
    assert rectangular_pinmesh == equal_rectangular_pinmesh
    assert rectangular_pinmesh != unequal_rectangular_pinmesh

def test_rectangular_pinmesh_hash(rectangular_pinmesh,
                                  equal_rectangular_pinmesh,
                                  unequal_rectangular_pinmesh):
    assert hash(rectangular_pinmesh) == hash(equal_rectangular_pinmesh)
    assert hash(rectangular_pinmesh) != hash(unequal_rectangular_pinmesh)

def test_rectangular_pinmesh_write_to_string(rectangular_pinmesh):
    output = rectangular_pinmesh.write_to_string(prefix="  ", mpact_ids={rectangular_pinmesh: 42})
    expected_output = "  pinmesh 42 rec 1.0 2.0 3.0 / 1.0 2.0 3.0 / 1.0 2.0 3.0 / 10 10 10 / 10 10 10 / 5 5 5\n"
    assert output == expected_output

def test_rectangular_pinmesh_overlay(rectangular_pinmesh, openmc_fuel_material, openmc_moderator_material, openmc_pin):

    fuel_area = pi*0.4**2
    box_area  = 1.0*1.0
    fuel_frac = fuel_area / box_area
    mod_frac  = 1.0 - fuel_frac
    offset    = (-1.5, -1.5, 0.0)

    pinmesh = rectangular_pinmesh
    F       = Material.from_openmc_material(openmc_fuel_material)
    M       = Material.from_openmc_material(openmc_moderator_material)
    H       = Material.mix_materials([F, M], [fuel_frac, mod_frac], Material.MixPolicy(percent_type='vo'))

    overlay_policy = PinMesh.OverlayPolicy(method="centroid", num_procs=2)
    materials = pinmesh.overlay(model=openmc_pin, offset=offset, overlay_policy=overlay_policy)
    assert len(materials) == pinmesh.number_of_material_regions
    expected_materials = [M, M, M,
                          M, F, M,
                          M, M, M] * 3

    assert all(materials_are_close(material, expected_material)
               for material, expected_material in zip(materials, expected_materials))

    overlay_policy = PinMesh.OverlayPolicy(method="homogenized", n_samples=100000, num_procs=2)
    materials = pinmesh.overlay(model=openmc_pin, offset=offset, overlay_policy=overlay_policy)
    expected_materials = [M, M, M,
                          M, H, M,
                          M, M, M] * 3

    assert all(materials_are_close(material, expected_material)
               for material, expected_material in zip(materials, expected_materials))

@pytest.fixture
def general_cylindrical_pinmesh():
    r = [0.5, 1.0, 1.5]
    xMin, xMax = -1.0, 1.0
    yMin, yMax = -1.0, 1.0
    zvals = [1.0, 2.0, 3.0]
    ndivr = [1, 2, 2]
    ndiva = [8, 8, 8, 8, 8, 8]
    ndivz = [5, 5, 5]
    return GeneralCylindricalPinMesh(r, xMin, xMax, yMin, yMax, zvals, ndivr, ndiva, ndivz)

@pytest.fixture
def equal_general_cylindrical_pinmesh():
    r = [0.5*(1+TOL), 1.0*(1-TOL), 1.5*(1+TOL)]
    xMin, xMax = -1.0*(1+TOL), 1.0*(1-TOL)
    yMin, yMax = -1.0*(1-TOL), 1.0*(1+TOL)
    zvals = [1.0*(1-TOL), 2.0*(1+TOL), 3.0*(1-TOL)]
    ndivr = [1, 2, 2]
    ndiva = [8, 8, 8, 8, 8, 8]
    ndivz = [5, 5, 5]
    return GeneralCylindricalPinMesh(r, xMin, xMax, yMin, yMax, zvals, ndivr, ndiva, ndivz)

@pytest.fixture
def unequal_general_cylindrical_pinmesh():
    r = [0.5, 1.0, 1.4]
    xMin, xMax = -1.0, 1.0
    yMin, yMax = -1.0, 1.0
    zvals = [1.0, 2.0, 3.0]
    ndivr = [1, 2, 2]
    ndiva = [8, 8, 8, 8, 8, 8]
    ndivz = [5, 5, 5]
    return GeneralCylindricalPinMesh(r, xMin, xMax, yMin, yMax, zvals, ndivr, ndiva, ndivz)

def test_general_cylindrical_pinmesh_initialization(general_cylindrical_pinmesh):
    pinmesh = general_cylindrical_pinmesh
    assert pinmesh.number_of_material_regions == 12
    assert pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6, 8, 9, 10]
    assert_allclose([pinmesh.pitch[i] for i in ['X','Y','Z']], [2., 2., 3.])
    assert isclose(pinmesh.xMin, -1.0)
    assert isclose(pinmesh.xMax,  1.0)
    assert isclose(pinmesh.yMin, -1.0)
    assert isclose(pinmesh.yMax,  1.0)
    assert_allclose(pinmesh.r,     [0.5, 1.0, 1.5])
    assert_allclose(pinmesh.zvals, [1.0, 2.0, 3.0])
    assert_allclose(pinmesh.ndivr, [1, 2, 2])
    assert_allclose(pinmesh.ndiva, [8, 8, 8, 8, 8, 8])
    assert_allclose(pinmesh.ndivz, [5, 5, 5])

def test_general_cylindrical_pinmesh_equality(general_cylindrical_pinmesh,
                                              equal_general_cylindrical_pinmesh,
                                              unequal_general_cylindrical_pinmesh):
        assert general_cylindrical_pinmesh == equal_general_cylindrical_pinmesh
        assert general_cylindrical_pinmesh != unequal_general_cylindrical_pinmesh

def test_general_cylindrical_pinmesh_hash(general_cylindrical_pinmesh,
                                          equal_general_cylindrical_pinmesh,
                                          unequal_general_cylindrical_pinmesh):
    assert hash(general_cylindrical_pinmesh) == hash(equal_general_cylindrical_pinmesh)
    assert hash(general_cylindrical_pinmesh) != hash(unequal_general_cylindrical_pinmesh)

def test_general_cylindrical_pinmesh_write_to_string(general_cylindrical_pinmesh):
    output = general_cylindrical_pinmesh.write_to_string(prefix="  ", mpact_ids={general_cylindrical_pinmesh: 42})
    expected_output = "  pinmesh 42 gcyl 0.5 1.0 / -1.0 1.0 -1.0 1.0 / 1.0 2.0 3.0 / 1 2 / 8 8 8 8 / 5 5 5\n"
    assert output == expected_output
