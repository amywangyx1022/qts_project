"""Configuration constants and parameter bounds for crypto spread trading."""

from typing import Final
from pathlib import Path

# Exchange configuration
EXCHANGES: Final[list[str]] = ["Binance", "Coinbase", "OKX"]

# Spread pairs (exchange_A, exchange_B) - spread = price_A - price_B
SPREAD_PAIRS: Final[list[tuple[str, str]]] = [
    ("Binance", "Coinbase"),
    ("Binance", "OKX"),
    ("Coinbase", "OKX"),
]

# EMA half-life in seconds (3 hours = 10800 seconds)
EMA_HALFLIFE_SECONDS: Final[int] = 3 * 60 * 60  # 10800 seconds

# Capital parameters
INITIAL_CAPITAL: Final[float] = 80_000.0  # USDT
STOP_CAPITAL: Final[float] = 40_000.0  # USDT - stop trading threshold
POSITION_SIZE_ETH: Final[float] = 1.0  # 1 ETH per trade

# Trading cost scenarios (zeta)
ZETA_VALUES: Final[list[float]] = [0.0, 0.0001]

# Parameter search bounds for Optuna
PARAM_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "j": (0.01, 5.0),  # exit band level
    "g": (0.1, 10.0),  # entry band level (must be > j)
    "l": (1.0, 20.0),  # stop-loss level (must be > g)
    "N": (1, 100),  # rank for persistent spread
    "M": (1, 1000),  # lookback window (must be >= N)
}

# Minimum trades per day threshold
MIN_TRADES_PER_DAY: Final[int] = 5

# Data directory (relative to week2)
DATA_DIR: Final[Path] = Path(__file__).parent.parent.parent / "data"

# Output directory
OUTPUT_DIR: Final[Path] = Path(__file__).parent.parent.parent / "outputs"

# Optimization settings
DEFAULT_N_TRIALS: Final[int] = 500
DEFAULT_N_JOBS: Final[int] = 1  # Parallel jobs for Optuna
RANDOM_SEED: Final[int] = 42

# Annualization factor for Sharpe ratio
# For 1-second data: seconds in a year (365 * 24 * 3600)
SECONDS_PER_YEAR: Final[int] = 365 * 24 * 3600  # 31,536,000
