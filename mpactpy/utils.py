from typing import List, Union, TypeVar, Callable, Any
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
