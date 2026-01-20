"""Backtesting engine for crypto spread trading."""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from .strategy import (
    TradingStrategy,
    Position,
    Trade,
    PositionSide,
    ExitReason,
)
from .signals import compute_all_signals
from .config import INITIAL_CAPITAL, STOP_CAPITAL, POSITION_SIZE_ETH


@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: list[Trade]
    equity_curve: pd.Series
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    trades_per_day: float
    num_days: int
    stop_loss_count: int
    params: dict = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine for crypto spread trading strategy.

    Implements:
    - Capital tracking with stop-trading threshold
    - Stop-loss with day pause
    - Position at end of data close
    - Trading cost application
    """

    def __init__(
        self,
        strategy: TradingStrategy,
        initial_capital: float = INITIAL_CAPITAL,
        stop_capital: float = STOP_CAPITAL,
        position_size: float = POSITION_SIZE_ETH,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Trading strategy instance
            initial_capital: Starting capital in USDT
            stop_capital: Stop trading if capital falls below this
            position_size: Position size in ETH
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.stop_capital = stop_capital
        self.position_size = position_size

    def run(
        self,
        timestamps: pd.Series,
        price_a: pd.Series,
        price_b: pd.Series,
        p_small: pd.Series,
        p_large: pd.Series,
    ) -> BacktestResult:
        """
        Run backtest on provided data.

        Args:
            timestamps: Timestamp series
            price_a: Exchange A prices
            price_b: Exchange B prices
            p_small: N-th smallest spread (p^S)
            p_large: N-th largest spread (p^L)

        Returns:
            BacktestResult with all metrics and trades
        """
        # Initialize state
        capital = self.initial_capital
        equity = [capital]
        trades: list[Trade] = []
        position: Optional[Position] = None
        stop_loss_count = 0

        # Day pause tracking (after stop loss)
        pause_until: Optional[pd.Timestamp] = None
        trading_stopped = False

        n = len(timestamps)

        for i in range(n):
            ts = timestamps.iloc[i]
            pa = price_a.iloc[i]
            pb = price_b.iloc[i]
            ps = p_small.iloc[i]
            pl = p_large.iloc[i]

            # Track mark-to-market equity FIRST (before any position changes)
            # This captures unrealized PnL during trades for proper drawdown calculation
            if position is not None:
                unrealized_pnl = self.strategy.calculate_trade_pnl(
                    position.side,
                    position.entry_price_a,
                    position.entry_price_b,
                    pa,
                    pb,
                    self.position_size,
                )
                equity.append(capital + unrealized_pnl)
            else:
                equity.append(capital)

            # Check if we've hit capital stop
            if capital <= self.stop_capital:
                trading_stopped = True

            # Check if in day pause
            if pause_until is not None and ts < pause_until:
                in_pause = True
            else:
                in_pause = False
                pause_until = None

            # Skip trading logic if stopped or in pause (but still track equity)
            if trading_stopped or in_pause:
                # If we have a position during capital stop, close it
                if trading_stopped and position is not None:
                    pnl = self.strategy.calculate_trade_pnl(
                        position.side,
                        position.entry_price_a,
                        position.entry_price_b,
                        pa,
                        pb,
                        self.position_size,
                    )
                    trade = Trade(
                        side=position.side,
                        entry_time=position.entry_time,
                        exit_time=ts,
                        entry_price_a=position.entry_price_a,
                        entry_price_b=position.entry_price_b,
                        exit_price_a=pa,
                        exit_price_b=pb,
                        entry_spread=ps if position.side == PositionSide.SHORT else pl,
                        exit_spread=ps if position.side == PositionSide.SHORT else pl,
                        pnl=pnl,
                        exit_reason=ExitReason.CAPITAL_STOP,
                        size_eth=self.position_size,
                    )
                    trades.append(trade)
                    capital += pnl
                    position = None
                # Equity already tracked at start of iteration
                continue

            # If we have a position, check for exit
            if position is not None:
                should_exit, exit_reason = self.strategy.check_exit_signal(
                    position, ps, pl
                )

                if should_exit:
                    # Calculate PnL and close position
                    pnl = self.strategy.calculate_trade_pnl(
                        position.side,
                        position.entry_price_a,
                        position.entry_price_b,
                        pa,
                        pb,
                        self.position_size,
                    )
                    trade = Trade(
                        side=position.side,
                        entry_time=position.entry_time,
                        exit_time=ts,
                        entry_price_a=position.entry_price_a,
                        entry_price_b=position.entry_price_b,
                        exit_price_a=pa,
                        exit_price_b=pb,
                        entry_spread=ps if position.side == PositionSide.SHORT else pl,
                        exit_spread=ps if position.side == PositionSide.SHORT else pl,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        size_eth=self.position_size,
                    )
                    trades.append(trade)
                    capital += pnl
                    position = None

                    # If stop loss, pause for remainder of day
                    if exit_reason == ExitReason.STOP_LOSS:
                        stop_loss_count += 1
                        # Pause until next day (normalize to midnight, add 1 day)
                        current_day = ts.normalize()
                        pause_until = current_day + pd.Timedelta(days=1)

            # If flat, check for entry
            if position is None:
                entry_signal = self.strategy.check_entry_signal(ps, pl)

                if entry_signal != PositionSide.FLAT:
                    position = Position(
                        side=entry_signal,
                        entry_price_a=pa,
                        entry_price_b=pb,
                        entry_time=ts,
                        size_eth=self.position_size,
                    )

            # Equity already tracked at start of iteration

        # Close any remaining position at end of data
        if position is not None:
            pa = price_a.iloc[-1]
            pb = price_b.iloc[-1]
            ps = p_small.iloc[-1]
            pl = p_large.iloc[-1]

            pnl = self.strategy.calculate_trade_pnl(
                position.side,
                position.entry_price_a,
                position.entry_price_b,
                pa,
                pb,
                self.position_size,
            )
            trade = Trade(
                side=position.side,
                entry_time=position.entry_time,
                exit_time=timestamps.iloc[-1],
                entry_price_a=position.entry_price_a,
                entry_price_b=position.entry_price_b,
                exit_price_a=pa,
                exit_price_b=pb,
                entry_spread=ps if position.side == PositionSide.SHORT else pl,
                exit_spread=ps if position.side == PositionSide.SHORT else pl,
                pnl=pnl,
                exit_reason=ExitReason.END_OF_DATA,
                size_eth=self.position_size,
            )
            trades.append(trade)
            capital += pnl
            equity[-1] = capital

        # Build equity curve
        equity_series = pd.Series(equity[1:], index=timestamps)

        # Calculate metrics
        from .metrics import (
            calculate_sharpe_ratio,
            calculate_max_drawdown,
            calculate_win_rate,
        )

        # Number of unique days in data
        num_days = len(timestamps.dt.date.unique())
        num_trades = len(trades)
        trades_per_day = num_trades / num_days if num_days > 0 else 0

        # Calculate Sharpe ratio using daily returns
        sharpe = calculate_sharpe_ratio(equity_series)
        max_dd = calculate_max_drawdown(equity_series)
        win_rate = calculate_win_rate(trades)

        total_return = (capital - self.initial_capital) / self.initial_capital

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            final_capital=capital,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=num_trades,
            trades_per_day=trades_per_day,
            num_days=num_days,
            stop_loss_count=stop_loss_count,
            params=self.strategy.get_params_dict(),
        )

        return result


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    timestamps: pd.Series,
    j: float,
    g: float,
    l: float,
    N: int,
    M: int,
    zeta: float = 0.0,
    initial_capital: float = INITIAL_CAPITAL,
    stop_capital: float = STOP_CAPITAL,
) -> BacktestResult:
    """
    Convenience function to run a full backtest with given parameters.

    Args:
        price_a: Exchange A prices
        price_b: Exchange B prices
        timestamps: Timestamp series
        j: Exit band level
        g: Entry band level
        l: Stop-loss level
        N: Rank for persistent spread
        M: Lookback window
        zeta: Trading cost parameter
        initial_capital: Starting capital
        stop_capital: Stop trading threshold

    Returns:
        BacktestResult
    """
    # Compute signals
    signals = compute_all_signals(price_a, price_b, N, M)

    # Create strategy
    strategy = TradingStrategy(j, g, l, N, M, zeta)

    # Create and run backtest engine
    engine = BacktestEngine(strategy, initial_capital, stop_capital)

    result = engine.run(
        timestamps,
        price_a,
        price_b,
        signals["p_small"],
        signals["p_large"],
    )

    return result
