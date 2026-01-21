"""Optuna optimization for crypto spread trading parameters."""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from .backtest import run_backtest
from .config import (
    PARAM_BOUNDS,
    DEFAULT_N_TRIALS,
    DEFAULT_N_JOBS,
    RANDOM_SEED,
    INITIAL_CAPITAL,
    STOP_CAPITAL,
    MIN_TRADES_PER_DAY,
)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    study: optuna.Study
    best_params: dict
    best_value: float
    spread_pair: tuple[str, str]
    zeta: float
    n_trials: int
    n_completed: int
    n_pruned: int


def create_objective(
    price_a: pd.Series,
    price_b: pd.Series,
    timestamps: pd.Series,
    zeta: float,
    initial_capital: float = INITIAL_CAPITAL,
    stop_capital: float = STOP_CAPITAL,
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for Optuna optimization.

    The objective maximizes Sharpe ratio while respecting constraints.

    Args:
        price_a: Exchange A prices
        price_b: Exchange B prices
        timestamps: Timestamp series
        zeta: Trading cost parameter
        initial_capital: Starting capital
        stop_capital: Stop trading threshold

    Returns:
        Objective function for Optuna
    """
    num_days = len(timestamps.dt.date.unique())

    def objective(trial: optuna.Trial) -> float:
        # Sample parameters with constraints
        # j: exit band level
        j = trial.suggest_float("j", PARAM_BOUNDS["j"][0], PARAM_BOUNDS["j"][1])

        # g: entry band level (must be > j)
        g_min = j + 0.01
        g_max = PARAM_BOUNDS["g"][1]
        if g_min >= g_max:
            # Invalid constraint, prune
            raise optuna.TrialPruned()
        g = trial.suggest_float("g", g_min, g_max)

        # l: stop-loss level (must be > g)
        l_min = g + 0.01
        l_max = PARAM_BOUNDS["l"][1]
        if l_min >= l_max:
            raise optuna.TrialPruned()
        l = trial.suggest_float("l", l_min, l_max)

        # N: rank for persistent spread
        N = trial.suggest_int("N", int(PARAM_BOUNDS["N"][0]), int(PARAM_BOUNDS["N"][1]))

        # M: lookback window (must be >= N)
        M_min = N
        M_max = int(PARAM_BOUNDS["M"][1])
        if M_min >= M_max:
            M = M_min
        else:
            M = trial.suggest_int("M", M_min, M_max)

        try:
            # Run backtest
            result = run_backtest(
                price_a=price_a,
                price_b=price_b,
                timestamps=timestamps,
                j=j,
                g=g,
                l=l,
                N=N,
                M=M,
                zeta=zeta,
                initial_capital=initial_capital,
                stop_capital=stop_capital,
            )

            # Filter out parameter combinations with insufficient trades
            # Per requirement: discard cases with fewer than 5 trades per day
            if result.trades_per_day < MIN_TRADES_PER_DAY:
                raise optuna.TrialPruned()

            # Return Sharpe ratio as objective
            sharpe = result.sharpe_ratio

            # Handle invalid values
            if np.isnan(sharpe) or np.isinf(sharpe):
                raise optuna.TrialPruned()

            # Store additional metrics as user attributes
            trial.set_user_attr("max_drawdown", result.max_drawdown)
            trial.set_user_attr("total_return", result.total_return)
            trial.set_user_attr("win_rate", result.win_rate)
            trial.set_user_attr("num_trades", result.num_trades)
            trial.set_user_attr("trades_per_day", result.trades_per_day)
            trial.set_user_attr("stop_loss_count", result.stop_loss_count)

            return sharpe

        except Exception as e:
            # Log error and prune
            trial.set_user_attr("error", str(e))
            raise optuna.TrialPruned()

    return objective


def run_optimization(
    price_a: pd.Series,
    price_b: pd.Series,
    timestamps: pd.Series,
    spread_pair: tuple[str, str],
    zeta: float,
    n_trials: int = DEFAULT_N_TRIALS,
    n_jobs: int = -1,  # Use all CPU cores for parallel optimization
    seed: int = RANDOM_SEED,
    show_progress: bool = True,
    storage: Optional[str] = None,
) -> OptimizationResult:
    """
    Run Optuna optimization for a spread pair.

    Args:
        price_a: Exchange A prices
        price_b: Exchange B prices
        timestamps: Timestamp series
        spread_pair: Tuple of (exchange_a, exchange_b)
        zeta: Trading cost parameter
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        seed: Random seed for reproducibility
        show_progress: Whether to show progress bar
        storage: Optional Optuna storage URL (for SQLite persistence)

    Returns:
        OptimizationResult with best parameters and study
    """
    # Create study name
    pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
    study_name = f"crypto_spread_{pair_name}_zeta{zeta}"

    # Create sampler and pruner
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=20,  # Random trials before TPE (reduced from 50)
        multivariate=True,    # Consider parameter correlations
        warn_independent_sampling=False,  # Suppress warnings for dynamic search space
    )
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
    )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize Sharpe ratio
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=False,
    )

    # Create objective function
    objective = create_objective(
        price_a=price_a,
        price_b=price_b,
        timestamps=timestamps,
        zeta=zeta,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress,
    )

    # Get results
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    result = OptimizationResult(
        study=study,
        best_params=study.best_params if n_completed > 0 else {},
        best_value=study.best_value if n_completed > 0 else float("-inf"),
        spread_pair=spread_pair,
        zeta=zeta,
        n_trials=n_trials,
        n_completed=n_completed,
        n_pruned=n_pruned,
    )

    return result


def get_best_trial_metrics(study: optuna.Study) -> dict:
    """
    Get all metrics from the best trial.

    Args:
        study: Completed Optuna study

    Returns:
        Dictionary with best trial parameters and metrics
    """
    if not study.best_trial:
        return {}

    best = study.best_trial
    metrics = {
        "sharpe_ratio": best.value,
        **best.params,
        **best.user_attrs,
    }
    return metrics


def trials_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """
    Convert study trials to a DataFrame for analysis.

    Args:
        study: Optuna study

    Returns:
        DataFrame with all trial results
    """
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        record = {
            "trial_number": trial.number,
            "sharpe_ratio": trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        records.append(record)

    return pd.DataFrame(records)


def print_optimization_summary(result: OptimizationResult) -> None:
    """
    Print a formatted summary of optimization results.

    Args:
        result: OptimizationResult from run_optimization
    """
    pair_name = f"{result.spread_pair[0]}-{result.spread_pair[1]}"

    print("=" * 80)
    print(f"Optimization Results: {pair_name}")
    print(f"Trading Cost (zeta): {result.zeta}")
    print("=" * 80)

    print(f"\nTrials: {result.n_completed} completed, {result.n_pruned} pruned")

    if result.n_completed > 0:
        print("\nBest Parameters:")
        for param, value in result.best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")

        print(f"\nBest Sharpe Ratio: {result.best_value:.4f}")

        # Get additional metrics
        metrics = get_best_trial_metrics(result.study)
        if "max_drawdown" in metrics:
            print(f"\nPerformance Metrics (best trial):")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  Total Return: {metrics.get('total_return', 0)*100:.4f}%")
            print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"  Trades per Day: {metrics.get('trades_per_day', 0):.1f}")
            print(f"  Stop Losses: {metrics.get('stop_loss_count', 0)}")
    else:
        print("\nNo completed trials - all pruned due to constraints.")

    print("=" * 80)
