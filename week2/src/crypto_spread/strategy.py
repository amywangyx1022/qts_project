"""Trading strategy implementation for crypto spread trading."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class PositionSide(Enum):
    """Position side enumeration."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(Enum):
    """Exit reason enumeration."""
    NONE = "NONE"
    NORMAL = "NORMAL"  # Exit band crossed
    STOP_LOSS = "STOP_LOSS"
    END_OF_DATA = "END_OF_DATA"
    CAPITAL_STOP = "CAPITAL_STOP"


@dataclass
class Position:
    """Represents a trading position."""
    side: PositionSide
    entry_price_a: float  # Exchange A price at entry
    entry_price_b: float  # Exchange B price at entry
    entry_time: pd.Timestamp
    size_eth: float = 1.0

    def __post_init__(self):
        if self.side == PositionSide.FLAT:
            raise ValueError("Cannot create a FLAT position object")


@dataclass
class Trade:
    """Represents a completed trade."""
    side: PositionSide
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    entry_spread: float  # p_small or p_large at entry
    exit_spread: float   # p_small or p_large at exit
    pnl: float
    exit_reason: ExitReason
    size_eth: float = 1.0


class TradingStrategy:
    """
    Crypto spread trading strategy implementation.

    Trading Rules (from PDF):
    - SHORT 1 ETH when p^S > g (spread expensive)
    - LONG 1 ETH when p^L < -g (spread cheap)
    - Exit SHORT when p^S < j OR p^S > l (stop loss)
    - Exit LONG when p^L > -j OR p^L < -l (stop loss)
    """

    def __init__(
        self,
        j: float,
        g: float,
        l: float,
        N: int,
        M: int,
        zeta: float = 0.0
    ):
        """
        Initialize strategy with parameters.

        Args:
            j: Exit band level
            g: Entry band level (should be > j)
            l: Stop-loss level (should be > g)
            N: Rank for persistent spread
            M: Lookback window (should be >= N)
            zeta: Trading cost parameter
        """
        self.j = j
        self.g = g
        self.l = l
        self.N = N
        self.M = M
        self.zeta = zeta

        # Validate constraints
        if g <= j:
            raise ValueError(f"g ({g}) must be > j ({j})")
        if l <= g:
            raise ValueError(f"l ({l}) must be > g ({g})")
        if M < N:
            raise ValueError(f"M ({M}) must be >= N ({N})")

    def check_entry_signal(
        self,
        p_small: float,
        p_large: float
    ) -> PositionSide:
        """
        Check if entry conditions are met.

        Args:
            p_small: N-th smallest spread (p^S)
            p_large: N-th largest spread (p^L)

        Returns:
            PositionSide.SHORT if p^S > g
            PositionSide.LONG if p^L < -g
            PositionSide.FLAT otherwise
        """
        if pd.isna(p_small) or pd.isna(p_large):
            return PositionSide.FLAT

        # SHORT when spread is expensive (p^S > g)
        if p_small > self.g:
            return PositionSide.SHORT

        # LONG when spread is cheap (p^L < -g)
        if p_large < -self.g:
            return PositionSide.LONG

        return PositionSide.FLAT

    def check_exit_signal(
        self,
        position: Position,
        p_small: float,
        p_large: float
    ) -> tuple[bool, ExitReason]:
        """
        Check if exit conditions are met for current position.

        Args:
            position: Current position
            p_small: N-th smallest spread (p^S)
            p_large: N-th largest spread (p^L)

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        if pd.isna(p_small) or pd.isna(p_large):
            return False, ExitReason.NONE

        if position.side == PositionSide.SHORT:
            # Exit SHORT when p^S < j (normal exit)
            if p_small < self.j:
                return True, ExitReason.NORMAL
            # Stop loss when p^S > l
            if p_small > self.l:
                return True, ExitReason.STOP_LOSS

        elif position.side == PositionSide.LONG:
            # Exit LONG when p^L > -j (normal exit)
            if p_large > -self.j:
                return True, ExitReason.NORMAL
            # Stop loss when p^L < -l
            if p_large < -self.l:
                return True, ExitReason.STOP_LOSS

        return False, ExitReason.NONE

    def calculate_trade_pnl(
        self,
        trade_side: PositionSide,
        entry_price_a: float,
        entry_price_b: float,
        exit_price_a: float,
        exit_price_b: float,
        size_eth: float = 1.0
    ) -> float:
        """
        Calculate PnL for a completed trade.

        For spread trading (price_A - price_B):
        - LONG spread: profit when spread increases (buy A, sell B)
        - SHORT spread: profit when spread decreases (sell A, buy B)

        Trading costs are applied on both entry and exit.

        Args:
            trade_side: LONG or SHORT
            entry_price_a: Exchange A price at entry
            entry_price_b: Exchange B price at entry
            exit_price_a: Exchange A price at exit
            exit_price_b: Exchange B price at exit
            size_eth: Position size in ETH

        Returns:
            Net PnL in USDT
        """
        # Calculate spread changes
        entry_spread = entry_price_a - entry_price_b
        exit_spread = exit_price_a - exit_price_b
        spread_change = exit_spread - entry_spread

        # PnL from spread movement
        if trade_side == PositionSide.LONG:
            # Long spread profits when spread increases
            raw_pnl = spread_change * size_eth
        elif trade_side == PositionSide.SHORT:
            # Short spread profits when spread decreases
            raw_pnl = -spread_change * size_eth
        else:
            raise ValueError(f"Invalid trade side: {trade_side}")

        # Trading costs
        # Cost = zeta * position_value (using average price as position value)
        avg_entry_price = (entry_price_a + entry_price_b) / 2
        avg_exit_price = (exit_price_a + exit_price_b) / 2
        entry_cost = self.zeta * avg_entry_price * size_eth
        exit_cost = self.zeta * avg_exit_price * size_eth
        total_cost = entry_cost + exit_cost

        net_pnl = raw_pnl - total_cost
        return net_pnl

    def get_params_dict(self) -> dict:
        """Return strategy parameters as dictionary."""
        return {
            "j": self.j,
            "g": self.g,
            "l": self.l,
            "N": self.N,
            "M": self.M,
            "zeta": self.zeta,
        }

    def __repr__(self) -> str:
        return (
            f"TradingStrategy(j={self.j:.3f}, g={self.g:.3f}, l={self.l:.3f}, "
            f"N={self.N}, M={self.M}, zeta={self.zeta})"
        )
