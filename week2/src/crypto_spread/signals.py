"""Spread signal calculations for crypto spread trading."""

import numpy as np
import pandas as pd
from numba import njit
from typing import Optional

from .config import EMA_HALFLIFE_SECONDS


def calculate_base_spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """
    Calculate base spread s^b = price_A - price_B.

    Args:
        price_a: Price series for exchange A
        price_b: Price series for exchange B

    Returns:
        Base spread series
    """
    return price_a - price_b


def calculate_ema_alpha(halflife_seconds: int) -> float:
    """
    Calculate EMA alpha from half-life.

    The half-life is the number of periods for the weight to decay to 0.5.
    alpha = 1 - exp(-ln(2) / halflife)

    Args:
        halflife_seconds: Half-life in seconds

    Returns:
        Alpha parameter for EMA
    """
    return 1 - np.exp(-np.log(2) / halflife_seconds)


def calculate_ema(
    series: pd.Series,
    halflife_seconds: int = EMA_HALFLIFE_SECONDS
) -> pd.Series:
    """
    Calculate exponential moving average with specified half-life.

    Args:
        series: Input series
        halflife_seconds: Half-life in seconds (default: 3 hours = 10800)

    Returns:
        EMA series
    """
    alpha = calculate_ema_alpha(halflife_seconds)
    # Pandas ewm with alpha parameter
    # span, com, and halflife are alternatives but alpha is most direct
    ema = series.ewm(alpha=alpha, adjust=False).mean()
    return ema


def calculate_shifted_spread(
    base_spread: pd.Series,
    ema: pd.Series
) -> pd.Series:
    """
    Calculate shifted (demeaned) spread s = s^b - a.

    Args:
        base_spread: Base spread series
        ema: EMA of base spread

    Returns:
        Shifted spread series
    """
    return base_spread - ema


@njit(cache=True)
def _rolling_nth_smallest(arr: np.ndarray, N: int, M: int) -> np.ndarray:
    """
    Numba-optimized rolling N-th smallest calculation.

    Args:
        arr: Input array
        N: Rank (1-indexed, so N=1 means smallest)
        M: Window size

    Returns:
        Array with N-th smallest values over rolling window
    """
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(M - 1, n):
        window = arr[i - M + 1 : i + 1].copy()
        window.sort()
        # N is 1-indexed, so N=1 gives index 0 (smallest)
        result[i] = window[N - 1]

    return result


@njit(cache=True)
def _rolling_nth_largest(arr: np.ndarray, N: int, M: int) -> np.ndarray:
    """
    Numba-optimized rolling N-th largest calculation.

    Args:
        arr: Input array
        N: Rank (1-indexed, so N=1 means largest)
        M: Window size

    Returns:
        Array with N-th largest values over rolling window
    """
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(M - 1, n):
        window = arr[i - M + 1 : i + 1].copy()
        window.sort()
        # N-th largest is at index M - N (after sorting ascending)
        result[i] = window[M - N]

    return result


def calculate_persistent_levels(
    shifted_spread: pd.Series,
    N: int,
    M: int
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate persistent spread levels p^S (N-th smallest) and p^L (N-th largest)
    over the most recent M observations.

    Args:
        shifted_spread: Shifted spread series
        N: Rank for persistent spread (N-th smallest/largest)
        M: Lookback window size (M >= N)

    Returns:
        Tuple of (p_small, p_large) series
    """
    if M < N:
        raise ValueError(f"M ({M}) must be >= N ({N})")

    arr = shifted_spread.values.astype(np.float64)

    # Calculate N-th smallest and N-th largest
    p_small = _rolling_nth_smallest(arr, N, M)
    p_large = _rolling_nth_largest(arr, N, M)

    # Convert back to Series
    p_small_series = pd.Series(p_small, index=shifted_spread.index, name="p_small")
    p_large_series = pd.Series(p_large, index=shifted_spread.index, name="p_large")

    return p_small_series, p_large_series


def compute_all_signals(
    price_a: pd.Series,
    price_b: pd.Series,
    N: int,
    M: int,
    halflife_seconds: int = EMA_HALFLIFE_SECONDS
) -> pd.DataFrame:
    """
    Compute all spread signals for a given parameter set.

    Args:
        price_a: Price series for exchange A
        price_b: Price series for exchange B
        N: Rank for persistent spread
        M: Lookback window size
        halflife_seconds: EMA half-life in seconds

    Returns:
        DataFrame with columns: base_spread, ema, shifted_spread, p_small, p_large
    """
    # Calculate base spread
    base_spread = calculate_base_spread(price_a, price_b)

    # Calculate EMA
    ema = calculate_ema(base_spread, halflife_seconds)

    # Calculate shifted spread
    shifted_spread = calculate_shifted_spread(base_spread, ema)

    # Calculate persistent levels
    p_small, p_large = calculate_persistent_levels(shifted_spread, N, M)

    # Combine into DataFrame
    signals = pd.DataFrame({
        "base_spread": base_spread,
        "ema": ema,
        "shifted_spread": shifted_spread,
        "p_small": p_small,
        "p_large": p_large,
    })

    return signals


if __name__ == "__main__":
    # Test signal calculations
    import numpy as np

    # Create sample data
    np.random.seed(42)
    n = 1000
    price_a = pd.Series(2500 + np.cumsum(np.random.randn(n) * 0.1))
    price_b = pd.Series(2500 + np.cumsum(np.random.randn(n) * 0.1))

    # Compute signals
    signals = compute_all_signals(price_a, price_b, N=5, M=60)

    print("Signal DataFrame:")
    print(signals.head(70))
    print("\nStatistics:")
    print(signals.describe())
