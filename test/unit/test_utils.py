import pytest
from math import isclose, pi, sqrt

from mpactpy.utils import (
    equal_arc_length_ndivs,
    equal_thickness_ndivs,
    equal_thickness_regions,
    list_to_str,
    num_to_str,
    relative_round,
    subdivide_ring,
)


def test_relative_round():
    assert relative_round(0.0) == 0.0
    assert relative_round(1.23456789) == 1.23457
    assert relative_round(1.23456789, rel_tol=1e-5) == 1.23457
    assert relative_round(1234.56789, rel_tol=1e-5) == 1234.57
    assert relative_round(-1.23456789, rel_tol=1e-5) == -1.23457

    close_values = [0.000499999999998835, 0.000500000000030809]
    assert relative_round(close_values[0]) == relative_round(close_values[1])
    assert relative_round(close_values[0]) == 0.0005

    with pytest.raises(AssertionError):
        relative_round(1.0, rel_tol=0.0)


def test_list_to_str():
    values = [0.000499999999998835, 0.000500000000030809]

    assert list_to_str(values) == "0.0005 0.0005"
    assert list_to_str([1.88514677844989], rel_tol=1e-9) == "1.885146778"


def test_num_to_str():
    assert num_to_str(0.000499999999998835) == num_to_str(0.000500000000030809)
    assert num_to_str(1.88514677844989) == "1.88515"
    assert num_to_str(1.88514677844989, rel_tol=1e-9) == "1.885146778"


def test_subdivide_ring():
    assert equal_thickness_regions(1.0, 3.0, 4) == [1.5, 2.0, 2.5, 3.0]

    radii = subdivide_ring(1.0, 3.0, 4, "equal_thickness")
    assert radii == [1.5, 2.0, 2.5, 3.0]

    radii = subdivide_ring(0.0, 2.0, 4, "equal_volume")
    expected = [1.0, sqrt(2.0), sqrt(3.0), 2.0]
    assert len(radii) == len(expected)
    assert all(isclose(radii[i], expected[i]) for i in range(len(expected)))

    radii = subdivide_ring(1.0, 3.0, 2, "equal_volume")
    expected = [sqrt(5.0), 3.0]
    assert len(radii) == len(expected)
    assert all(isclose(radii[i], expected[i]) for i in range(len(expected)))

    ring_area_1 = radii[0] * radii[0] - 1.0
    ring_area_2 = radii[1] * radii[1] - radii[0] * radii[0]
    assert isclose(ring_area_1, ring_area_2)

    with pytest.raises(AssertionError):
        subdivide_ring(-1.0, 1.0, 1, "equal_volume")

    with pytest.raises(AssertionError):
        subdivide_ring(1.0, 1.0, 1, "equal_volume")

    with pytest.raises(AssertionError):
        subdivide_ring(0.0, 1.0, 0, "equal_volume")

    with pytest.raises(AssertionError):
        subdivide_ring(0.0, 1.0, 1, "bad")


def test_equal_thickness_ndivs():
    assert equal_thickness_ndivs([1.0], 0.5) == [2]
    assert equal_thickness_ndivs([1.1], 0.5) == [3]
    assert equal_thickness_ndivs([1.0], 2.0) == [1]
    assert equal_thickness_ndivs([1.0, 1.25, 1.75], 0.5) == [2, 3, 4]
    assert equal_thickness_ndivs([1.0, 1.0], 0.5) == [2, 2]

    with pytest.raises(AssertionError):
        equal_thickness_ndivs([0.0], 0.5)

    with pytest.raises(AssertionError):
        equal_thickness_ndivs([], 0.5)

    with pytest.raises(AssertionError):
        equal_thickness_ndivs([1.0, 0.0], 0.5)


def test_equal_arc_length_ndivs():
    assert equal_arc_length_ndivs([1.0], 2.0 * pi) == [1]
    assert equal_arc_length_ndivs([1.0], 2.0) == [4]
    assert equal_arc_length_ndivs([0.0], 1.0) == [1]
    assert equal_arc_length_ndivs([1.0], 2.0, multiple_of=4) == [4]
    assert equal_arc_length_ndivs([1.0], 1.0, multiple_of=4) == [8]
    assert equal_arc_length_ndivs([0.0, 1.0, 2.0], 2.0) == [1, 4, 7]
    assert equal_arc_length_ndivs([1.0, 2.0], 2.0, multiple_of=4) == [4, 8]

    with pytest.raises(AssertionError):
        equal_arc_length_ndivs([-1.0], 1.0)

    with pytest.raises(AssertionError):
        equal_arc_length_ndivs([], 1.0)

    with pytest.raises(AssertionError):
        equal_arc_length_ndivs([1.0, -1.0], 1.0)
