from typing import List, Union, TypeVar, Callable, Any, Literal
from collections.abc import Hashable
from decimal import Decimal, ROUND_HALF_UP
import math
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import openmc

# The relative tolerance for rounding floating point numbers
ROUNDING_RELATIVE_TOLERANCE = 1E-5

# Avogadro's number
AVOGADRO = openmc.data.AVOGADRO

# Room Temperature in Kelvin
ROOM_TEMPERATURE = 293.6

RadialDivisionType = Literal["equal_thickness", "equal_volume"]


def equal_thickness_regions(lower_bound: float,
                            upper_bound: float,
                            num_div:     int) -> List[float]:
    """Return region upper bounds that divide an interval into equal thicknesses.

    Parameters
    ----------
    lower_bound : float
        Lower bound of the full interval.
    upper_bound : float
        Upper bound of the full interval.
    num_div : int
        Number of equal-thickness regions.

    Returns
    -------
    List[float]
        Upper bounds for each equal-thickness region, ordered from lower to upper.
    """
    assert upper_bound > lower_bound, f"upper_bound = {upper_bound}, lower_bound = {lower_bound}"
    assert num_div > 0, f"num_div = {num_div}"

    thickness = upper_bound - lower_bound
    return [lower_bound + thickness * i / num_div for i in range(1, num_div + 1)]


def equal_thickness_ndivs(thicknesses:      List[float],
                          target_thickness: float) -> List[int]:
    """Return equal-thickness division counts for interval thicknesses.

    Parameters
    ----------
    thicknesses : List[float]
        Thicknesses of the intervals to subdivide.
    target_thickness : float
        Maximum target thickness of each equal-thickness division.

    Returns
    -------
    List[int]
        Number of equal-thickness divisions for each interval.
    """
    def ndiv(thickness: float) -> int:
        assert thickness > 0.0, f"thickness = {thickness}"
        assert target_thickness > 0.0, f"target_thickness = {target_thickness}"
        return max(1, math.ceil(thickness / target_thickness))

    assert thicknesses, "thicknesses must not be empty"
    return [ndiv(thickness) for thickness in thicknesses]


def equal_arc_length_ndivs(radii:             List[float],
                           target_arc_length: float,
                           multiple_of:       int = 1) -> List[int]:
    """Return equal-angle division counts for radii and target arc length.

    Parameters
    ----------
    radii : List[float]
        Radii at which arc lengths are evaluated.
    target_arc_length : float
        Maximum target arc length of each equal-angle division.
    multiple_of : int
        Optional multiple to round each division count up to. Defaults to 1.

    Returns
    -------
    List[int]
        Number of equal-angle divisions for each radius.
    """
    def ndiv(radius: float) -> int:
        assert radius >= 0.0, f"radius = {radius}"
        assert target_arc_length > 0.0, f"target_arc_length = {target_arc_length}"
        assert multiple_of > 0, f"multiple_of = {multiple_of}"

        num_div = max(1, math.ceil(2.0 * math.pi * radius / target_arc_length))
        return math.ceil(num_div / multiple_of) * multiple_of

    assert radii, "radii must not be empty"
    return [ndiv(radius) for radius in radii]


def equal_volume_ring_radii(inner_radius: float,
                            outer_radius: float,
                            num_div:      int) -> List[float]:
    """Return ring outer radii that divide an annulus into equal areas.

    Parameters
    ----------
    inner_radius : float
        Inner radius of the full annulus.
    outer_radius : float
        Outer radius of the full annulus.
    num_div : int
        Number of equal-area radial regions.

    Returns
    -------
    List[float]
        Outer radii for each equal-area ring, ordered from inner to outer.
    """
    assert inner_radius >= 0.0, f"inner_radius = {inner_radius}"
    assert outer_radius > inner_radius, f"outer_radius = {outer_radius}, inner_radius = {inner_radius}"
    assert num_div > 0, f"num_div = {num_div}"

    inner_area = inner_radius * inner_radius
    area_step  = (outer_radius * outer_radius - inner_area) / num_div
    return [math.sqrt(inner_area + i * area_step) for i in range(1, num_div + 1)]


def subdivide_ring(inner_radius: float,
                   outer_radius: float,
                   num_div:      int,
                   div_type:     RadialDivisionType) -> List[float]:
    """Return ring outer radii using the requested subdivision rule.

    Parameters
    ----------
    inner_radius : float
        Inner radius of the full annulus.
    outer_radius : float
        Outer radius of the full annulus.
    num_div : int
        Number of radial regions.
    div_type : RadialDivisionType
        Rule used to place the radial interfaces.

    Returns
    -------
    List[float]
        Outer radii for each subdivided ring, ordered from inner to outer.
    """
    assert div_type in ("equal_thickness", "equal_volume"), f"div_type = {div_type}"
    if div_type == "equal_thickness":
        assert inner_radius >= 0.0, f"inner_radius = {inner_radius}"
        return equal_thickness_regions(inner_radius, outer_radius, num_div)
    return equal_volume_ring_radii(inner_radius, outer_radius, num_div)


def relative_round(value: float, rel_tol: float = ROUNDING_RELATIVE_TOLERANCE) -> float:
    """ Rounds a floating-point number to a precision consistent with a given relative tolerance.

    Parameters:
    -----------
    value : float
        The number to round.
    rel_tol : float, optional
        The relative tolerance for rounding.

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
             rtol: float = ROUNDING_RELATIVE_TOLERANCE,
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


def num_to_str(num:          Union[float, int],
               print_length: int = None,
               rel_tol:      float = ROUNDING_RELATIVE_TOLERANCE) -> str:
    """Convert a numerical value to a canonical MPACT input string.

    Floating point values are rounded before writing so values that are equal
    within the writer tolerance produce identical text in MPACT cards.

    Parameters
    ----------
    num : float or int
        The number to convert to a string.
    print_length : int
        The print spacing for the string.
    rel_tol : float
        The relative tolerance for rounding floating point values.

    Returns
    -------
    str
        The number as a string.
    """

    if isinstance(num, float):
        num = relative_round(num, rel_tol)
        if math.isclose(num, round(num)):
            return f"{num:.1f}" if print_length is None else f"{num:{print_length}.1f}"
        return f"{num:.15g}" if print_length is None else f"{num:{print_length}.15g}"
    return f"{str(num)}" if print_length is None else f"{str(num):{print_length}}"


def list_to_str(input_list:   List[Union[float, int]],
                print_length: int = None,
                rel_tol:      float = ROUNDING_RELATIVE_TOLERANCE) -> str:
    """ Converts a list of numerical values to an equally spaced string

    Parameters
    ----------
    input_list : List[Union[float, int]]
        The list to be converted to a string
    print_length : int
        The print spacing for the string
    rel_tol : float
        The relative tolerance for rounding floating point values.

    Returns
    -------
    str
        The list as a string
    """

    return ' '.join(num_to_str(x, print_length, rel_tol) for x in input_list)

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

def atomic_mass(name: str) -> float:
    """ Return the atomic mass of a nuclide or element.

    If an isotope is provided (e.g., 'U235', 'H1'), returns the exact atomic mass.
    If an element is provided (e.g., 'U', 'H'), returns the natural-abundance-weighted
    average atomic mass.

    Parameters
    ----------
    name : str
        Isotope (e.g., 'U235') or element (e.g., 'U')

    Returns
    -------
    float
        Atomic mass in g/mol (amu)

    Raises
    ------
    ValueError
        If the element/isotope is not recognized or lacks abundance data.
    """

    name = name.strip()

    try:
        return openmc.data.atomic_mass(name)
    except KeyError:
        pass

    try:
        return sum(abundance * openmc.data.atomic_mass(isotope)
                   for isotope, abundance in openmc.data.isotopes(name))
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Cannot find atomic mass for '{name}'.") from exc


@contextmanager
def temporary_environment(var: str, value: str):
    """ Context manager for temporarily setting environment variables

    Parameters
    ----------
    var : str
        The name of the environment variable to set.
    value : str
        The temporary value to assign to the environment variable.
    """
    original = os.environ.get(var)
    os.environ[var] = value
    try:
        yield
    finally:
        if original is not None:
            os.environ[var] = original
        else:
            del os.environ[var]

S = TypeVar('S')
R = TypeVar('R')
def process_parallel_work(work_items:      List[S],
                          worker_function: Callable[..., R],
                          num_processes:   int,
                          *worker_args:    Any) -> List[R]:
    """Process work items in parallel using chunked distribution.

    Parameters
    ----------
    work_items : List[S]
        List of work items to process
    worker_function : Callable
        Function to process each chunk. Should accept (chunk, *worker_args)
        and return a list of results
    num_processes : int
        Maximum number of processes to use
    *worker_args : Any
        Additional arguments to pass to the worker function

    Returns
    -------
    List[R]
        Results in the same order as input work_items
    """
    if not work_items:
        return []

    if num_processes <= 1:
        # Process in serial
        return worker_function(work_items, *worker_args)

    # Process in parallel with chunking
    chunk_indices = np.array_split(range(len(work_items)), num_processes)
    work_chunks = [[work_items[i] for i in indices] for indices in chunk_indices if len(indices) > 0]

    with ProcessPoolExecutor(max_workers=min(num_processes, len(work_chunks))) as executor:
        future_to_chunk_index = {
            executor.submit(worker_function, chunk, *worker_args): i
            for i, chunk in enumerate(work_chunks)
        }

        chunk_results = [None] * len(work_chunks)
        for future in as_completed(future_to_chunk_index):
            chunk_index = future_to_chunk_index[future]
            chunk_results[chunk_index] = future.result()

    # Flatten results maintaining order
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)

    return results
