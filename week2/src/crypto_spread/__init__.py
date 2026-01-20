"""Crypto Spread Trading Optimization Package."""

from .config import (
    EXCHANGES,
    SPREAD_PAIRS,
    EMA_HALFLIFE_SECONDS,
    INITIAL_CAPITAL,
    STOP_CAPITAL,
    POSITION_SIZE_ETH,
    ZETA_VALUES,
    PARAM_BOUNDS,
    MIN_TRADES_PER_DAY,
)

__all__ = [
    "EXCHANGES",
    "SPREAD_PAIRS",
    "EMA_HALFLIFE_SECONDS",
    "INITIAL_CAPITAL",
    "STOP_CAPITAL",
    "POSITION_SIZE_ETH",
    "ZETA_VALUES",
    "PARAM_BOUNDS",
    "MIN_TRADES_PER_DAY",
]
