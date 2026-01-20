"""Visualization functions for crypto spread trading analysis."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import optuna

if TYPE_CHECKING:
    from .strategy import Trade
    from .backtest import BacktestResult

from .config import OUTPUT_DIR


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def plot_spread_signals(
    timestamps: pd.Series,
    base_spread: pd.Series,
    ema: pd.Series,
    shifted_spread: pd.Series,
    p_small: pd.Series,
    p_large: pd.Series,
    spread_pair: tuple[str, str],
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot spread signals over time.

    Args:
        timestamps: Timestamp series
        base_spread: Base spread (price_A - price_B)
        ema: EMA of base spread
        shifted_spread: Shifted spread (demeaned)
        p_small: N-th smallest persistent level
        p_large: N-th largest persistent level
        spread_pair: Tuple of exchange names
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    pair_name = f"{spread_pair[0]} - {spread_pair[1]}"

    # Top plot: Base spread and EMA
    ax1 = axes[0]
    ax1.plot(timestamps, base_spread, label="Base Spread", alpha=0.7, linewidth=0.5)
    ax1.plot(timestamps, ema, label="EMA (3hr halflife)", color="red", linewidth=1)
    ax1.set_ylabel("Spread (USDT)")
    ax1.set_title(f"{pair_name}: Base Spread and EMA")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Shifted spread and persistent levels
    ax2 = axes[1]
    ax2.plot(timestamps, shifted_spread, label="Shifted Spread", alpha=0.7, linewidth=0.5)
    ax2.plot(timestamps, p_small, label="p^S (N-th smallest)", color="green", linewidth=1)
    ax2.plot(timestamps, p_large, label="p^L (N-th largest)", color="orange", linewidth=1)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Shifted Spread (USDT)")
    ax2.set_title(f"{pair_name}: Shifted Spread and Persistent Levels")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_equity_curve(
    equity_curve: pd.Series,
    trades: list["Trade"],
    spread_pair: tuple[str, str],
    zeta: float,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot equity curve with trade markers.

    Args:
        equity_curve: Equity series over time
        trades: List of completed trades
        spread_pair: Tuple of exchange names
        zeta: Trading cost parameter
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    pair_name = f"{spread_pair[0]} - {spread_pair[1]}"

    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, linewidth=1, color="blue", label="Equity")

    # Mark trades
    for trade in trades:
        color = "green" if trade.pnl > 0 else "red"
        ax.axvline(x=trade.entry_time, color=color, alpha=0.3, linewidth=0.5)

    # Add drawdown shading
    running_max = equity_curve.cummax()
    ax.fill_between(
        equity_curve.index,
        equity_curve,
        running_max,
        alpha=0.3,
        color="red",
        label="Drawdown",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Capital (USDT)")
    ax.set_title(f"{pair_name}: Equity Curve (zeta={zeta})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_trade_distribution(
    trades: list["Trade"],
    spread_pair: tuple[str, str],
    zeta: float,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot distribution of trade PnLs.

    Args:
        trades: List of completed trades
        spread_pair: Tuple of exchange names
        zeta: Trading cost parameter
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pair_name = f"{spread_pair[0]} - {spread_pair[1]}"
    pnls = [t.pnl for t in trades]

    # Left: Histogram
    ax1 = axes[0]
    ax1.hist(pnls, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax1.axvline(x=np.mean(pnls), color="green", linestyle="-", linewidth=1, label=f"Mean: {np.mean(pnls):.2f}")
    ax1.set_xlabel("PnL (USDT)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"{pair_name}: Trade PnL Distribution (zeta={zeta})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Cumulative PnL
    ax2 = axes[1]
    cumulative_pnl = np.cumsum(pnls)
    ax2.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl, linewidth=1)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Trade Number")
    ax2.set_ylabel("Cumulative PnL (USDT)")
    ax2.set_title(f"{pair_name}: Cumulative PnL (zeta={zeta})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_optimization_history(
    study: optuna.Study,
    spread_pair: tuple[str, str],
    zeta: float,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot optimization convergence history.

    Args:
        study: Completed Optuna study
        spread_pair: Tuple of exchange names
        zeta: Trading cost parameter
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pair_name = f"{spread_pair[0]} - {spread_pair[1]}"

    # Get completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials to plot")
        return fig

    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]

    # Left: Trial values over time
    ax1 = axes[0]
    ax1.scatter(trial_numbers, values, alpha=0.5, s=10)
    # Rolling best
    best_so_far = np.maximum.accumulate(values)
    ax1.plot(trial_numbers, best_so_far, color="red", linewidth=2, label="Best so far")
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title(f"{pair_name}: Optimization History (zeta={zeta})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Value distribution
    ax2 = axes[1]
    ax2.hist(values, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(x=study.best_value, color="red", linestyle="--", linewidth=2, label=f"Best: {study.best_value:.4f}")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{pair_name}: Sharpe Ratio Distribution (zeta={zeta})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_parameter_importance(
    study: optuna.Study,
    spread_pair: tuple[str, str],
    zeta: float,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot parameter importance from Optuna study.

    Args:
        study: Completed Optuna study
        spread_pair: Tuple of exchange names
        zeta: Trading cost parameter
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    pair_name = f"{spread_pair[0]} - {spread_pair[1]}"

    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        values = list(importance.values())

        ax.barh(params, values, color="steelblue", edgecolor="black")
        ax.set_xlabel("Importance")
        ax.set_title(f"{pair_name}: Parameter Importance (zeta={zeta})")
        ax.grid(True, alpha=0.3, axis="x")

    except Exception as e:
        ax.text(0.5, 0.5, f"Could not compute importance:\n{str(e)}",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_parameter_heatmap(
    study: optuna.Study,
    param_x: str,
    param_y: str,
    spread_pair: tuple[str, str],
    zeta: float,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 2D heatmap of parameter values vs Sharpe ratio.

    Args:
        study: Completed Optuna study
        param_x: Parameter for x-axis
        param_y: Parameter for y-axis
        spread_pair: Tuple of exchange names
        zeta: Trading cost parameter
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    pair_name = f"{spread_pair[0]} - {spread_pair[1]}"

    # Get completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials to plot")
        return fig

    x_vals = [t.params.get(param_x) for t in trials]
    y_vals = [t.params.get(param_y) for t in trials]
    sharpe_vals = [t.value for t in trials]

    scatter = ax.scatter(x_vals, y_vals, c=sharpe_vals, cmap="RdYlGn", s=20, alpha=0.7)
    plt.colorbar(scatter, label="Sharpe Ratio")

    # Mark best trial
    best_idx = np.argmax(sharpe_vals)
    ax.scatter([x_vals[best_idx]], [y_vals[best_idx]], color="blue", s=100,
               marker="*", edgecolor="black", linewidth=1, label="Best", zorder=5)

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"{pair_name}: {param_x} vs {param_y} (zeta={zeta})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def generate_summary_report(
    results: dict,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a text summary report of all optimization results.

    Args:
        results: Dictionary mapping (spread_pair, zeta) to BacktestResult
        output_path: Optional path to save report

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CRYPTO SPREAD TRADING OPTIMIZATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    for key, result in results.items():
        spread_pair, zeta = key
        pair_name = f"{spread_pair[0]} - {spread_pair[1]}"

        lines.append(f"Spread Pair: {pair_name}")
        lines.append(f"Trading Cost (zeta): {zeta}")
        lines.append("-" * 40)

        if hasattr(result, "best_params") and result.best_params:
            lines.append("Best Parameters:")
            for param, value in result.best_params.items():
                if isinstance(value, float):
                    lines.append(f"  {param}: {value:.4f}")
                else:
                    lines.append(f"  {param}: {value}")

            lines.append(f"\nBest Sharpe Ratio: {result.best_value:.4f}")
            lines.append(f"Completed Trials: {result.n_completed}")
            lines.append(f"Pruned Trials: {result.n_pruned}")
        else:
            lines.append("No successful optimization results.")

        lines.append("")
        lines.append("=" * 80)
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Saved report: {output_path}")

    return report


def save_all_plots(
    study: optuna.Study,
    backtest_result: "BacktestResult",
    timestamps: pd.Series,
    signals_df: pd.DataFrame,
    spread_pair: tuple[str, str],
    zeta: float,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Save all visualization plots for a single optimization run.

    Args:
        study: Completed Optuna study
        backtest_result: Backtest result with best parameters
        timestamps: Timestamp series
        signals_df: DataFrame with signal columns
        spread_pair: Tuple of exchange names
        zeta: Trading cost parameter
        output_dir: Output directory for plots
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    pair_name = f"{spread_pair[0]}_{spread_pair[1]}"
    zeta_str = f"zeta{zeta}".replace(".", "p")

    # 1. Spread signals
    plot_spread_signals(
        timestamps=timestamps,
        base_spread=signals_df["base_spread"],
        ema=signals_df["ema"],
        shifted_spread=signals_df["shifted_spread"],
        p_small=signals_df["p_small"],
        p_large=signals_df["p_large"],
        spread_pair=spread_pair,
        output_path=output_dir / f"spread_signals_{pair_name}.png",
        show=False,
    )

    # 2. Equity curve
    plot_equity_curve(
        equity_curve=backtest_result.equity_curve,
        trades=backtest_result.trades,
        spread_pair=spread_pair,
        zeta=zeta,
        output_path=output_dir / f"equity_curve_{pair_name}_{zeta_str}.png",
        show=False,
    )

    # 3. Trade distribution
    if backtest_result.trades:
        plot_trade_distribution(
            trades=backtest_result.trades,
            spread_pair=spread_pair,
            zeta=zeta,
            output_path=output_dir / f"trade_distribution_{pair_name}_{zeta_str}.png",
            show=False,
        )

    # 4. Optimization history
    plot_optimization_history(
        study=study,
        spread_pair=spread_pair,
        zeta=zeta,
        output_path=output_dir / f"optimization_history_{pair_name}_{zeta_str}.png",
        show=False,
    )

    # 5. Parameter importance
    plot_parameter_importance(
        study=study,
        spread_pair=spread_pair,
        zeta=zeta,
        output_path=output_dir / f"parameter_importance_{pair_name}_{zeta_str}.png",
        show=False,
    )

    # 6. Parameter heatmaps
    param_pairs = [("j", "g"), ("g", "l"), ("N", "M")]
    for px, py in param_pairs:
        plot_parameter_heatmap(
            study=study,
            param_x=px,
            param_y=py,
            spread_pair=spread_pair,
            zeta=zeta,
            output_path=output_dir / f"heatmap_{px}_{py}_{pair_name}_{zeta_str}.png",
            show=False,
        )

    plt.close("all")
    print(f"All plots saved to {output_dir}")
