import pytest
from math import isclose
from numpy.testing import assert_allclose
from mpactpy.utils import ROUNDING_RELATIVE_TOLERANCE
from mpactpy.pinmesh import RectangularPinMesh, GeneralCylindricalPinMesh

TOL = ROUNDING_RELATIVE_TOLERANCE * 1E-1

@pytest.fixture
def rectangular_pinmesh():
    xvals = [1.0, 2.0, 3.0]
    yvals = [1.0, 2.0, 3.0]
    zvals = [1.0, 2.0, 3.0]
    ndivx = [10, 10, 10]
    ndivy = [10, 10, 10]
    ndivz = [5, 5, 5]
    return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz, mpact_id=1)

@pytest.fixture
def equal_rectangular_pinmesh():
    xvals = [1.0*(1+TOL), 2.0*(1-TOL), 3.0*(1+TOL)]
    yvals = [1.0*(1-TOL), 2.0*(1+TOL), 3.0*(1+TOL)]
    zvals = [1.0*(1+TOL), 2.0*(1+TOL), 3.0*(1-TOL)]
    ndivx = [10, 10, 10]
    ndivy = [10, 10, 10]
    ndivz = [5, 5, 5]
    return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz, mpact_id=1)

@pytest.fixture
def unequal_rectangular_pinmesh():
    xvals = [1.0, 2.0, 5.0]
    yvals = [1.0, 2.0, 3.0]
    zvals = [1.0, 2.0, 3.0]
    ndivx = [10, 10, 10]
    ndivy = [10, 10, 10]
    ndivz = [5, 5, 5]
    return RectangularPinMesh(xvals, yvals, zvals, ndivx, ndivy, ndivz, mpact_id=1)

def test_rectangular_pinmesh_initialization(rectangular_pinmesh):
    pinmesh = rectangular_pinmesh
    assert pinmesh.mpact_id == 1
    assert pinmesh.number_of_material_regions == 27
    assert pinmesh.regions_inside_bounds == 27
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
    output = rectangular_pinmesh.write_to_string(prefix="test ")
    expected_output = "test pinmesh 1 rec 1.0 2.0 3.0 / 1.0 2.0 3.0 / 1.0 2.0 3.0 / 10 10 10 / 10 10 10 / 5 5 5\n"
    assert output == expected_output


@pytest.fixture
def general_cylindrical_pinmesh():
    r = [0.5, 1.0, 1.5]
    xMin, xMax = -1.0, 1.0
    yMin, yMax = -1.0, 1.0
    zvals = [1.0, 2.0, 3.0]
    ndivr = [1, 2, 2]
    ndiva = [8, 8, 8, 8, 8, 8]
    ndivz = [5, 5, 5]
    return GeneralCylindricalPinMesh(r, xMin, xMax, yMin, yMax, zvals, ndivr, ndiva, ndivz, mpact_id=1)

@pytest.fixture
def equal_general_cylindrical_pinmesh():
    r = [0.5*(1+TOL), 1.0*(1-TOL), 1.5*(1+TOL)]
    xMin, xMax = -1.0*(1+TOL), 1.0*(1-TOL)
    yMin, yMax = -1.0*(1-TOL), 1.0*(1+TOL)
    zvals = [1.0*(1-TOL), 2.0*(1+TOL), 3.0*(1-TOL)]
    ndivr = [1, 2, 2]
    ndiva = [8, 8, 8, 8, 8, 8]
    ndivz = [5, 5, 5]
    return GeneralCylindricalPinMesh(r, xMin, xMax, yMin, yMax, zvals, ndivr, ndiva, ndivz, mpact_id=1)

@pytest.fixture
def unequal_general_cylindrical_pinmesh():
    r = [0.5, 1.0, 1.4]
    xMin, xMax = -1.0, 1.0
    yMin, yMax = -1.0, 1.0
    zvals = [1.0, 2.0, 3.0]
    ndivr = [1, 2, 2]
    ndiva = [8, 8, 8, 8, 8, 8]
    ndivz = [5, 5, 5]
    return GeneralCylindricalPinMesh(r, xMin, xMax, yMin, yMax, zvals, ndivr, ndiva, ndivz, mpact_id=1)

def test_general_cylindrical_pinmesh_initialization(general_cylindrical_pinmesh):
    pinmesh = general_cylindrical_pinmesh
    assert pinmesh.mpact_id == 1
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
    output = general_cylindrical_pinmesh.write_to_string(prefix="test ")
    expected_output = "test pinmesh 1 gcyl 0.5 1.0 / -1.0 1.0 -1.0 1.0 / 1.0 2.0 3.0 / 1 2 / 8 8 8 8 / 5 5 5\n"
    assert output == expected_output