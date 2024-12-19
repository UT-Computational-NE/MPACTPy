from decimal import Decimal, ROUND_HALF_UP
import math

def relative_round(value: float, rel_tol: float =1e-9) -> float:
    """
    Rounds a floating-point number to a precision consistent with a given relative tolerance.

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

    abs_tol       = rel_tol * max(abs(value), 1.0)
    decimals      = max(0, int(math.ceil(-math.log10(abs_tol))))
    quantization  = Decimal(f'1e-{decimals}')
    rounded_value = Decimal(value).quantize(quantization, rounding=ROUND_HALF_UP)
    return float(rounded_value)
