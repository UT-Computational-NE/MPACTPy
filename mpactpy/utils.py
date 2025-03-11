from typing import List, Union, TypeVar
from collections.abc import Hashable
from decimal import Decimal, ROUND_HALF_UP
import math

import numpy as np

# The relative tolerance for rounding floating point numbers
ROUNDING_RELATIVE_TOLERANCE = 1E-5

def relative_round(value: float, rel_tol: float =1e-9) -> float:
    """ Rounds a floating-point number to a precision consistent with a given relative tolerance.

    Parameters:
    -----------
    value : float
        The number to round.
    rel_tol : float, optional
        The relative tolerance for rounding. Default is 1e-9.

    Returns:
    --------
    float
        The rounded value as a float.

    Notes:
    ------
    - Dynamically adjusts the number of decimal places based on the relative tolerance
      and the magnitude of the value.
    - The rounding ensures consistency with comparisons using math.isclose with the same rel_tol.
    """
    assert rel_tol > 0.

    if value == 0:
        return 0.0

    abs_tol       = rel_tol * abs(value)
    decimals      = max(0, int(math.ceil(-math.log10(abs_tol))))
    quantization  = Decimal(f'1e-{decimals}')
    rounded_value = Decimal(value).quantize(quantization, rounding=ROUND_HALF_UP)
    return float(rounded_value)


def allclose(rhs:  List[Union[float, int]],
             lhs:  List[Union[float, int]],
             rtol: float = 1E-05,
             atol: float = 1E-08) -> bool:
    """ Checks to see if the lists are approximately equal

    We need this helper function because np.allclose does not
    gracefully handle lists with different sizes.

    Parameters
    ----------
    rhs : List[Union[float, int]]
        The right-hand-side list to be compared
    lhs : List[Union[float, int]]
        The left-hand-side list to be compared
    rtol : float
        The relative tolerance for the comparison
    atol : float
        The absolute tolerance for the comparison

    Returns
    -------
    bool
        True if lists are element-wise approximately equal, False otherwise
    """

    if len(rhs) != len(lhs):
        return False
    return np.allclose(rhs, lhs, rtol, atol)


def list_to_str(input_list: List[Union[float, int]], print_length: int = None) -> str:
    """ Converts a list of numerical values to an equally spaced string

    Parameters
    ----------
    input_list : List[Union[float, int]]
        The list to be converted to a string
    print_length : int
        The print spacing for the string

    Returns
    -------
    str
        The list as a string
    """

    def print_num(num: Union[float, int], print_length: int) -> str:
        if isinstance(num, float):
            if math.isclose(num, round(num)):
                return f"{num:.1f}" if print_length is None else f"{num:{print_length}.1f}"
            return f"{num:.15g}" if print_length is None else f"{num:{print_length}.15g}"
        return f"{str(num)}" if print_length is None else f"{str(num):{print_length}}"

    return ' '.join(print_num(x, print_length) for x in input_list)

T = TypeVar('T', bound=Hashable)

def unique(elements: List[T]) -> List[T]:
    """ Function for extracting the unique elements of a list while preserving the original order of the elements

    Parameters
    ----------
    elements : List[T]
        The list of elements from which the unique elements will be identified

    Returns
    -------
    List[T]
        The list of unique elements
    """
    return list(dict.fromkeys(elements))


def is_rectangular(map_2D: List[List[T]]) -> bool:
    """ A helper function for checking whether or not a 2D map of elements is rectangular or not

    Parameters
    ----------
    map_2D : List[List[T]]
        The 2D Map to be checked

    Returns
    -------
    True if the 2D Map is rectangular, False otherwise
    """

    return bool(map_2D and map_2D[0]) and all(len(row) == len(map_2D[0]) for row in map_2D)
