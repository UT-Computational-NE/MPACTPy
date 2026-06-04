import pytest
from math import isclose, pi

from numpy.testing import assert_allclose
import openmc

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

    return geometry


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

    return (isclose(lhs.density, rhs.density, rel_tol=rel_tol)         and
            isclose(lhs.temperature, rhs.temperature, rel_tol=rel_tol) and
            lhs.replace_isotopes == rhs.replace_isotopes               and
            lhs.is_fluid                    == rhs.is_fluid            and
            lhs.is_depletable               == rhs.is_depletable       and
            lhs.has_resonance               == rhs.has_resonance       and
            lhs.is_fuel                     == rhs.is_fuel             and
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

def test_rectangular_pinmesh_subdivide(rectangular_pinmesh):
    subdivisions = RectangularPinMesh.Subdivisions(subd_x=[2, 1, 1], subd_y=[1, 2, 1], subd_z=[1, 1, 2])
    pinmesh, material_map = rectangular_pinmesh.subdivide(subdivisions)

    assert_allclose(pinmesh.xvals, [0.5, 1.0, 2.0, 3.0])
    assert_allclose(pinmesh.yvals, [1.0, 1.5, 2.0, 3.0])
    assert_allclose(pinmesh.zvals, [1.0, 2.0, 2.5, 3.0])
    assert pinmesh.ndivx == [10, 10, 10, 10]
    assert pinmesh.ndivy == [10, 10, 10, 10]
    assert pinmesh.ndivz == [5, 5, 5, 5]
    assert pinmesh.number_of_material_regions == 64

    expected_material_map = [z * 9 + y * 3 + x
                             for z in [0, 1, 2, 2]
                             for y in [0, 1, 1, 2]
                             for x in [0, 0, 1, 2]]
    assert material_map == expected_material_map

def test_rectangular_pinmesh_divide_into_quadrants(rectangular_pinmesh):
    quadrants = rectangular_pinmesh.divide_into_quadrants()
    [[nw, ne], [sw, se]] = quadrants

    def expected_material_map(x_indices, y_indices):
        return [z * 9 + y * 3 + x
                for z in range(3)
                for y in y_indices
                for x in x_indices]

    assert_allclose(nw[0].xvals, [1.0, 1.5])
    assert_allclose(nw[0].yvals, [0.5, 1.5])
    assert nw[0].ndivx == [10, 10]
    assert nw[0].ndivy == [10, 10]
    assert nw[0].ndivz == [5, 5, 5]
    assert nw[1] == expected_material_map([0, 1], [1, 2])

    assert_allclose(ne[0].xvals, [0.5, 1.5])
    assert_allclose(ne[0].yvals, [0.5, 1.5])
    assert ne[1] == expected_material_map([1, 2], [1, 2])

    assert_allclose(sw[0].xvals, [1.0, 1.5])
    assert_allclose(sw[0].yvals, [1.0, 1.5])
    assert sw[1] == expected_material_map([0, 1], [0, 1])

    assert_allclose(se[0].xvals, [0.5, 1.5])
    assert_allclose(se[0].yvals, [1.0, 1.5])
    assert se[1] == expected_material_map([1, 2], [0, 1])

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

    overlay_policy = PinMesh.OverlayPolicy(method="centroid", num_procs=1)
    materials = pinmesh.overlay(geometry=openmc_pin, offset=offset, overlay_policy=overlay_policy)
    assert len(materials) == pinmesh.number_of_material_regions
    expected_materials = [M, M, M,
                          M, F, M,
                          M, M, M] * 3

    assert all(materials_are_close(material, expected_material)
               for material, expected_material in zip(materials, expected_materials))

    overlay_policy = PinMesh.OverlayPolicy(method="homogenized", n_samples=100000, num_procs=1)
    materials = pinmesh.overlay(geometry=openmc_pin, offset=offset, overlay_policy=overlay_policy)
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

def test_general_cylindrical_pinmesh_subdivide(general_cylindrical_pinmesh):
    subdivisions = GeneralCylindricalPinMesh.Subdivisions(subd_r=[2, 1, 1, 2], subd_z=[1, 2, 1])
    pinmesh, material_map = general_cylindrical_pinmesh.subdivide(subdivisions)

    assert_allclose(pinmesh.r, [0.25, 0.5, 1.0, 1.5])
    assert_allclose(pinmesh.zvals, [1.0, 1.5, 2.0, 3.0])
    assert pinmesh.ndivr == [1, 1, 2, 2]
    assert pinmesh.ndiva == [8, 8, 8, 8, 8, 8, 8]
    assert pinmesh.ndivz == [5, 5, 5, 5]
    assert pinmesh.number_of_material_regions == 20
    assert material_map == [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11]

def test_general_cylindrical_pinmesh_divide_into_quadrants(general_cylindrical_pinmesh):
    quadrants = general_cylindrical_pinmesh.divide_into_quadrants()
    expected_bounds = [[(-1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)],
                       [(-1.0, 0.0, -1.0, 0.0), (0.0, 1.0, -1.0, 0.0)]]
    expected_material_map = list(range(general_cylindrical_pinmesh.number_of_material_regions))

    for row, bounds_row in zip(quadrants, expected_bounds):
        for (pinmesh, material_map), bounds in zip(row, bounds_row):
            assert (pinmesh.xMin, pinmesh.xMax, pinmesh.yMin, pinmesh.yMax) == bounds
            assert_allclose(pinmesh.r, general_cylindrical_pinmesh.r)
            assert_allclose(pinmesh.zvals, general_cylindrical_pinmesh.zvals)
            assert pinmesh.ndivr == general_cylindrical_pinmesh.ndivr
            assert pinmesh.ndiva == general_cylindrical_pinmesh.ndiva
            assert pinmesh.ndivz == general_cylindrical_pinmesh.ndivz
            assert material_map == expected_material_map
