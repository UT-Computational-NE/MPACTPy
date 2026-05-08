import pytest

from mpactpy.utils import list_to_str, num_to_str, relative_round


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
