"""Performance metrics calculations for crypto spread trading."""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .strategy import Trade

from .config import SECONDS_PER_YEAR


def calculate_sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,  # 24 hours × 365 days
) -> float:
    """
    Calculate annualized Sharpe ratio using hourly returns.

    Formula: Sharpe = sqrt(periods_per_year) * mean(excess_returns) / std(excess_returns)

    Args:
        equity_curve: Series of equity values with datetime index
        risk_free_rate: Annual risk-free rate (default: 0%)
        periods_per_year: Annualization factor (default: 8760 hours per year)

    Returns:
        Annualized Sharpe ratio
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    # Resample to hourly (end-of-hour values) and calculate returns
    hourly_equity = equity_curve.resample('H').last().dropna()

    if len(hourly_equity) < 2:
        return 0.0

    hourly_returns = hourly_equity.pct_change().dropna()

    if len(hourly_returns) < 1:
        return 0.0

    # Calculate excess returns
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = hourly_returns - rf_per_period

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    sharpe = np.sqrt(periods_per_year) * mean_excess / std_excess
    return float(sharpe)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown as a percentage.

    Args:
        equity_curve: Series of equity values over time

    Returns:
        Maximum drawdown as a decimal (e.g., 0.10 for 10% drawdown)
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Maximum drawdown (most negative value)
    max_dd = drawdown.min()

    return abs(float(max_dd))


def calculate_win_rate(trades: list["Trade"]) -> float:
    """
    Calculate win rate (percentage of profitable trades).

    Args:
        trades: List of completed trades

    Returns:
        Win rate as a decimal (e.g., 0.60 for 60% win rate)
    """
    if not trades:
        return 0.0

    profitable = sum(1 for t in trades if t.pnl > 0)
    return profitable / len(trades)


def calculate_profit_factor(trades: list["Trade"]) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        trades: List of completed trades

    Returns:
        Profit factor (>1 is profitable)
    """
    if not trades:
        return 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_average_trade_pnl(trades: list["Trade"]) -> float:
    """
    Calculate average PnL per trade.

    Args:
        trades: List of completed trades

    Returns:
        Average PnL per trade in USDT
    """
    if not trades:
        return 0.0

    return sum(t.pnl for t in trades) / len(trades)


def calculate_total_pnl(trades: list["Trade"]) -> float:
    """
    Calculate total PnL across all trades.

    Args:
        trades: List of completed trades

    Returns:
        Total PnL in USDT
    """
    return sum(t.pnl for t in trades)


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        total_return: Total return as decimal
        max_drawdown: Maximum drawdown as decimal

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float("inf") if total_return > 0 else 0.0

    return total_return / max_drawdown


def summarize_backtest_metrics(
    trades: list["Trade"],
    equity_curve: pd.Series,
    initial_capital: float,
    final_capital: float,
    num_days: int,
) -> dict:
    """
    Generate a comprehensive summary of backtest metrics.

    Args:
        trades: List of completed trades
        equity_curve: Equity curve series
        initial_capital: Starting capital
        final_capital: Ending capital
        num_days: Number of trading days

    Returns:
        Dictionary of metrics
    """
    # Calculate returns for Sharpe
    equity_arr = equity_curve.values
    returns = np.diff(equity_arr) / equity_arr[:-1]
    returns = pd.Series(returns)

    total_return = (final_capital - initial_capital) / initial_capital
    max_dd = calculate_max_drawdown(equity_curve)

    return {
        "num_trades": len(trades),
        "trades_per_day": len(trades) / num_days if num_days > 0 else 0,
        "total_pnl": calculate_total_pnl(trades),
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100,
        "win_rate": calculate_win_rate(trades),
        "win_rate_pct": calculate_win_rate(trades) * 100,
        "profit_factor": calculate_profit_factor(trades),
        "avg_trade_pnl": calculate_average_trade_pnl(trades),
        "calmar_ratio": calculate_calmar_ratio(total_return, max_dd),
        "final_capital": final_capital,
    }
