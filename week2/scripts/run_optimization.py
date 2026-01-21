#!/usr/bin/env python
"""
Main script to run crypto spread trading optimization.

Usage:
    uv run python scripts/run_optimization.py
    uv run python scripts/run_optimization.py --n-trials 100
    uv run python scripts/run_optimization.py --pairs Binance-Coinbase --zeta 0.0
    uv run python scripts/run_optimization.py --parallel  # Run pairs in parallel
"""

import argparse
import sys
from pathlib import Path
import json
import pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_spread.config import (
    SPREAD_PAIRS,
    ZETA_VALUES,
    DEFAULT_N_TRIALS,
    OUTPUT_DIR,
    DATA_DIR,
)
from crypto_spread.data_loader import (
    load_and_align_all_data,
    get_spread_pair_data,
)
from crypto_spread.signals import compute_all_signals
from crypto_spread.backtest import run_backtest
from crypto_spread.optimizer import (
    run_optimization,
    print_optimization_summary,
    get_best_trial_metrics,
    trials_to_dataframe,
)
from crypto_spread.visualization import (
    save_all_plots,
    generate_summary_report,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run crypto spread trading optimization"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of optimization trials (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Spread pairs to optimize (e.g., Binance-Coinbase Binance-OKX)",
    )
    parser.add_argument(
        "--zeta",
        nargs="+",
        type=float,
        default=None,
        help="Trading cost values (e.g., 0.0 0.0001)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-pickle",
        action="store_true",
        help="Save full results (including BacktestResult with trades) as pickle file",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run optimizations for different pairs in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )
    return parser.parse_args()


def run_single_optimization(
    spread_pair: tuple,
    zeta: float,
    aligned_data: pd.DataFrame,
    n_trials: int,
    seed: int,
    no_plots: bool,
    run_id: str,
) -> dict:
    """
    Run optimization for a single pair/zeta combination.

    Returns dict with results for this configuration.
    """
    pair_name = f"{spread_pair[0]}-{spread_pair[1]}"

    try:
        pair_data = get_spread_pair_data(aligned_data, spread_pair[0], spread_pair[1])
    except ValueError as e:
        print(f"[{run_id}] Skipping {pair_name}: {e}")
        return None

    timestamps = pair_data["ts"]
    price_a = pair_data["price_a"]
    price_b = pair_data["price_b"]

    print(f"[{run_id}] Starting: {pair_name}, zeta={zeta}")

    # Run optimization
    opt_result = run_optimization(
        price_a=price_a,
        price_b=price_b,
        timestamps=timestamps,
        spread_pair=spread_pair,
        zeta=zeta,
        n_trials=n_trials,
        seed=seed,
        show_progress=False,  # Disable progress bar in parallel mode
    )

    # Extract serializable data (Study object can't cross process boundary)
    if opt_result.n_completed > 0:
        metrics = get_best_trial_metrics(opt_result.study)
        best_params = opt_result.best_params
    else:
        metrics = {}
        best_params = {}

    result = {
        "spread_pair": spread_pair,
        "zeta": zeta,
        "best_params": best_params,
        "best_value": opt_result.best_value,
        "n_completed": opt_result.n_completed,
        "n_pruned": opt_result.n_pruned,
        "metrics": metrics,
        "backtest_result": None,
    }

    # Run backtest with best parameters if we have them
    if opt_result.n_completed > 0:
        backtest_result = run_backtest(
            price_a=price_a,
            price_b=price_b,
            timestamps=timestamps,
            j=best_params["j"],
            g=best_params["g"],
            l=best_params["l"],
            N=best_params["N"],
            M=best_params["M"],
            zeta=zeta,
        )
        result["backtest_result"] = backtest_result

        # Generate plots if requested
        if not no_plots:
            signals = compute_all_signals(
                price_a, price_b, best_params["N"], best_params["M"]
            )
            save_all_plots(
                study=opt_result.study,
                backtest_result=backtest_result,
                timestamps=timestamps,
                signals_df=signals,
                spread_pair=spread_pair,
                zeta=zeta,
                output_dir=OUTPUT_DIR / "plots",
            )

    sharpe_str = f"{opt_result.best_value:.4f}" if opt_result.n_completed > 0 else "N/A"
    print(f"[{run_id}] Completed: {pair_name}, zeta={zeta} (Sharpe: {sharpe_str})")

    return result


def run_seeded_optimization_worker(
    spread_pair: tuple,
    zeta: float,
    seed_params: dict,
    aligned_data: pd.DataFrame,
    n_trials: int,
    run_id: str,
) -> dict:
    """Run seeded optimization for a single configuration."""
    import optuna
    from crypto_spread.config import MIN_TRADES_PER_DAY

    pair_name = f"{spread_pair[0]}-{spread_pair[1]}"

    try:
        pair_data = get_spread_pair_data(aligned_data, spread_pair[0], spread_pair[1])
    except ValueError as e:
        print(f"[{run_id}] Skipping {pair_name}: {e}")
        return None

    def objective(trial):
        j = trial.suggest_float("j", max(0.01, seed_params['j'] * 0.5), seed_params['j'] * 2.0)
        g_min = j + 0.01
        g = trial.suggest_float("g", max(g_min, seed_params['g'] * 0.5), seed_params['g'] * 2.0)
        l_min = g + 0.01
        l = trial.suggest_float("l", max(l_min, seed_params['l'] * 0.5), seed_params['l'] * 2.0)
        N = trial.suggest_int("N", max(1, seed_params['N'] - 30), min(100, seed_params['N'] + 30))
        M = trial.suggest_int("M", max(N, seed_params['M'] - 100), min(1000, seed_params['M'] + 100))

        try:
            result = run_backtest(
                price_a=pair_data['price_a'],
                price_b=pair_data['price_b'],
                timestamps=pair_data['ts'],
                j=j, g=g, l=l, N=N, M=M,
                zeta=zeta,
            )
            if result.trades_per_day < MIN_TRADES_PER_DAY:
                raise optuna.TrialPruned()
            trial.set_user_attr("max_drawdown", result.max_drawdown)
            trial.set_user_attr("total_return", result.total_return)
            trial.set_user_attr("win_rate", result.win_rate)
            trial.set_user_attr("trades_per_day", result.trades_per_day)
            trial.set_user_attr("stop_loss_count", result.stop_loss_count)
            return result.sharpe_ratio
        except:
            raise optuna.TrialPruned()

    print(f"[{run_id}] Seeded search: {pair_name}, zeta={zeta}")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed == 0:
        return None

    best_params = study.best_params
    backtest_result = run_backtest(
        price_a=pair_data['price_a'],
        price_b=pair_data['price_b'],
        timestamps=pair_data['ts'],
        **best_params,
        zeta=zeta,
    )

    return {
        "spread_pair": spread_pair,
        "zeta": zeta,
        "best_params": best_params,
        "best_value": study.best_value,
        "n_completed": n_completed,
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "metrics": {
            "total_return": backtest_result.total_return,
            "max_drawdown": backtest_result.max_drawdown,
            "win_rate": backtest_result.win_rate,
            "trades_per_day": backtest_result.trades_per_day,
            "stop_loss_count": backtest_result.stop_loss_count,
        },
        "backtest_result": backtest_result,
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directories
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CRYPTO SPREAD TRADING OPTIMIZATION")
    print("=" * 80)

    # Load and align data
    print("\n[1/4] Loading and aligning data...")
    aligned_data = load_and_align_all_data(DATA_DIR)

    # Determine which pairs to optimize
    if args.pairs:
        pairs = []
        for p in args.pairs:
            parts = p.split("-")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
            else:
                print(f"Warning: Invalid pair format '{p}', skipping")
    else:
        pairs = SPREAD_PAIRS

    # Determine zeta values
    zeta_values = args.zeta if args.zeta else ZETA_VALUES

    print(f"\nSpread pairs: {pairs}")
    print(f"Zeta values: {zeta_values}")
    print(f"Trials per optimization: {args.n_trials}")
    print(f"Parallel mode: {args.parallel}")

    # Store all results
    all_results = {}
    all_backtest_results = {}

    # Build list of all configurations to run
    configs = []
    for spread_pair in pairs:
        for zeta in zeta_values:
            configs.append((spread_pair, zeta))

    total_runs = len(configs)
    print(f"Total configurations: {total_runs}")

    # Run optimization for each pair and zeta
    print("\n[2/4] Running optimizations...")

    if args.parallel and total_runs > 1:
        # Parallel execution
        n_workers = args.workers or min(mp.cpu_count(), total_runs)
        print(f"Using {n_workers} parallel workers")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for i, (spread_pair, zeta) in enumerate(configs, 1):
                run_id = f"{i}/{total_runs}"
                future = executor.submit(
                    run_single_optimization,
                    spread_pair,
                    zeta,
                    aligned_data,
                    args.n_trials,
                    args.seed,
                    args.no_plots,
                    run_id,
                )
                futures[future] = (spread_pair, zeta)

            for future in as_completed(futures):
                spread_pair, zeta = futures[future]
                try:
                    result = future.result()
                    if result:
                        key = (result["spread_pair"], result["zeta"])
                        # Store serialized optimization data (dict format for parallel mode)
                        all_results[key] = {
                            "best_params": result["best_params"],
                            "best_value": result["best_value"],
                            "n_completed": result["n_completed"],
                            "n_pruned": result["n_pruned"],
                            "metrics": result["metrics"],
                        }
                        if result["backtest_result"]:
                            all_backtest_results[key] = result["backtest_result"]
                except Exception as e:
                    pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
                    print(f"ERROR processing {pair_name}, zeta={zeta}: {e}")
    else:
        # Sequential execution
        for run_count, (spread_pair, zeta) in enumerate(configs, 1):
            pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
            print(f"\n--- Run {run_count}/{total_runs}: {pair_name}, zeta={zeta} ---")

            try:
                pair_data = get_spread_pair_data(aligned_data, spread_pair[0], spread_pair[1])
            except ValueError as e:
                print(f"Skipping {spread_pair}: {e}")
                continue

            timestamps = pair_data["ts"]
            price_a = pair_data["price_a"]
            price_b = pair_data["price_b"]

            # Run optimization
            opt_result = run_optimization(
                price_a=price_a,
                price_b=price_b,
                timestamps=timestamps,
                spread_pair=spread_pair,
                zeta=zeta,
                n_trials=args.n_trials,
                seed=args.seed,
                show_progress=True,
            )

            # Store results
            all_results[(spread_pair, zeta)] = opt_result

            # Print summary
            print_optimization_summary(opt_result)

            # Run backtest with best parameters if we have them
            if opt_result.n_completed > 0:
                best_params = opt_result.best_params
                backtest_result = run_backtest(
                    price_a=price_a,
                    price_b=price_b,
                    timestamps=timestamps,
                    j=best_params["j"],
                    g=best_params["g"],
                    l=best_params["l"],
                    N=best_params["N"],
                    M=best_params["M"],
                    zeta=zeta,
                )
                all_backtest_results[(spread_pair, zeta)] = backtest_result

                # Generate plots if requested
                if not args.no_plots:
                    signals = compute_all_signals(
                        price_a, price_b, best_params["N"], best_params["M"]
                    )
                    save_all_plots(
                        study=opt_result.study,
                        backtest_result=backtest_result,
                        timestamps=timestamps,
                        signals_df=signals,
                        spread_pair=spread_pair,
                        zeta=zeta,
                        output_dir=OUTPUT_DIR / "plots",
                    )

    # Phase 2: Seeded re-optimization if one zeta found better params
    print("\n[2.5/4] Seeded re-optimization for inconsistent results...")

    def is_dict_result(result):
        return isinstance(result, dict)

    seeded_configs = []
    for spread_pair in pairs:
        pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
        result_0 = all_results.get((spread_pair, 0.0))
        result_1 = all_results.get((spread_pair, 0.0001))

        if result_0 is None or result_1 is None:
            continue

        sharpe_0 = (result_0["best_value"] if is_dict_result(result_0) else result_0.best_value) if (result_0.get("n_completed", 0) if is_dict_result(result_0) else result_0.n_completed) > 0 else float('-inf')
        sharpe_1 = (result_1["best_value"] if is_dict_result(result_1) else result_1.best_value) if (result_1.get("n_completed", 0) if is_dict_result(result_1) else result_1.n_completed) > 0 else float('-inf')

        if sharpe_1 > sharpe_0:
            seed_params = result_1["best_params"] if is_dict_result(result_1) else result_1.best_params
            seeded_configs.append((spread_pair, 0.0, seed_params))
            print(f"  {pair_name}: zeta=0.0001 found better params ({sharpe_1:.2f} vs {sharpe_0:.2f}), will re-search")

    if seeded_configs:
        if args.parallel and len(seeded_configs) > 1:
            n_workers = args.workers or min(mp.cpu_count(), len(seeded_configs))
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for i, (spread_pair, zeta, seed_params) in enumerate(seeded_configs, 1):
                    run_id = f"seeded-{i}/{len(seeded_configs)}"
                    future = executor.submit(
                        run_seeded_optimization_worker,
                        spread_pair, zeta, seed_params, aligned_data, 30, run_id
                    )
                    futures[future] = (spread_pair, zeta)

                for future in as_completed(futures):
                    spread_pair, zeta = futures[future]
                    try:
                        result = future.result()
                        if result:
                            key = (result["spread_pair"], result["zeta"])
                            all_results[key] = {
                                "best_params": result["best_params"],
                                "best_value": result["best_value"],
                                "n_completed": result["n_completed"],
                                "n_pruned": result["n_pruned"],
                                "metrics": result["metrics"],
                            }
                            if result["backtest_result"]:
                                all_backtest_results[key] = result["backtest_result"]
                    except Exception as e:
                        print(f"ERROR in seeded optimization: {e}")
        else:
            for spread_pair, zeta, seed_params in seeded_configs:
                result = run_seeded_optimization_worker(
                    spread_pair, zeta, seed_params, aligned_data, 30,
                    f"seeded-{spread_pair[0]}-{spread_pair[1]}"
                )
                if result:
                    key = (result["spread_pair"], result["zeta"])
                    all_results[key] = {
                        "best_params": result["best_params"],
                        "best_value": result["best_value"],
                        "n_completed": result["n_completed"],
                        "n_pruned": result["n_pruned"],
                        "metrics": result["metrics"],
                    }
                    if result["backtest_result"]:
                        all_backtest_results[key] = result["backtest_result"]

    # Save results
    print("\n[3/4] Saving results...")

    # Generate and save summary report
    # Note: generate_summary_report expects OptimizationResult objects, so we skip it for parallel mode dicts
    if all_results and not is_dict_result(next(iter(all_results.values()))):
        report = generate_summary_report(
            all_results,
            output_path=OUTPUT_DIR / "results" / "optimization_summary.txt",
        )
        print(report)
    else:
        # For parallel mode, generate a simple summary
        summary_lines = ["=" * 60, "OPTIMIZATION SUMMARY (Parallel Mode)", "=" * 60, ""]
        for key, result in all_results.items():
            spread_pair, zeta = key
            pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
            if result["n_completed"] > 0:
                summary_lines.append(f"{pair_name} (zeta={zeta}): Sharpe={result['best_value']:.4f}")
            else:
                summary_lines.append(f"{pair_name} (zeta={zeta}): No completed trials")
        summary_lines.append("=" * 60)
        report = "\n".join(summary_lines)
        with open(OUTPUT_DIR / "results" / "optimization_summary.txt", "w") as f:
            f.write(report)
        print(report)

    # Save detailed results as JSON
    results_json = {}
    for key, opt_result in all_results.items():
        spread_pair, zeta = key
        pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
        key_str = f"{pair_name}_zeta{zeta}"

        if is_dict_result(opt_result):
            # Parallel mode: already a dict
            if opt_result["n_completed"] > 0:
                results_json[key_str] = {
                    "spread_pair": list(spread_pair),
                    "zeta": zeta,
                    "best_params": opt_result["best_params"],
                    "best_sharpe": opt_result["best_value"],
                    "n_completed": opt_result["n_completed"],
                    "n_pruned": opt_result["n_pruned"],
                    "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in opt_result["metrics"].items()},
                }
        else:
            # Sequential mode: OptimizationResult object
            if opt_result.n_completed > 0:
                metrics = get_best_trial_metrics(opt_result.study)
                results_json[key_str] = {
                    "spread_pair": list(spread_pair),
                    "zeta": zeta,
                    "best_params": opt_result.best_params,
                    "best_sharpe": opt_result.best_value,
                    "n_completed": opt_result.n_completed,
                    "n_pruned": opt_result.n_pruned,
                    "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                }

    with open(OUTPUT_DIR / "results" / "optimization_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'results' / 'optimization_results.json'}")

    # Save full results as pickle if requested
    if args.save_pickle:
        pickle_data = {
            "optimization_results": all_results,
            "backtest_results": all_backtest_results,
        }
        pickle_path = OUTPUT_DIR / "results" / "optimization_results.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(pickle_data, f)
        print(f"Saved: {pickle_path}")

    # Save trials dataframe for each study (only available in sequential mode)
    for key, opt_result in all_results.items():
        spread_pair, zeta = key
        pair_name = f"{spread_pair[0]}_{spread_pair[1]}"
        zeta_str = f"zeta{zeta}".replace(".", "p")

        # Skip if parallel mode (dict format - no Study object available)
        if is_dict_result(opt_result):
            continue

        trials_df = trials_to_dataframe(opt_result.study)
        if not trials_df.empty:
            csv_path = OUTPUT_DIR / "results" / f"trials_{pair_name}_{zeta_str}.csv"
            trials_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    print("\n[4/4] Optimization complete!")
    print(f"\nResults saved to: {OUTPUT_DIR}")

    # Print final comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"{'Spread Pair':<25} {'Zeta':<10} {'Sharpe':<10} {'Return %':<10} {'Trades/Day':<12}")
    print("-" * 80)

    for key, opt_result in all_results.items():
        spread_pair, zeta = key
        pair_name = f"{spread_pair[0]}-{spread_pair[1]}"

        if is_dict_result(opt_result):
            # Parallel mode: dict format
            if opt_result["n_completed"] > 0:
                metrics = opt_result["metrics"]
                sharpe = opt_result["best_value"]
                ret = metrics.get("total_return", 0) * 100
                tpd = metrics.get("trades_per_day", 0)
                print(f"{pair_name:<25} {zeta:<10.4f} {sharpe:<10.4f} {ret:<10.4f} {tpd:<12.1f}")
            else:
                print(f"{pair_name:<25} {zeta:<10.4f} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
        else:
            # Sequential mode: OptimizationResult object
            if opt_result.n_completed > 0:
                metrics = get_best_trial_metrics(opt_result.study)
                sharpe = opt_result.best_value
                ret = metrics.get("total_return", 0) * 100
                tpd = metrics.get("trades_per_day", 0)
                print(f"{pair_name:<25} {zeta:<10.4f} {sharpe:<10.4f} {ret:<10.4f} {tpd:<12.1f}")
            else:
                print(f"{pair_name:<25} {zeta:<10.4f} {'N/A':<10} {'N/A':<10} {'N/A':<12}")

    print("=" * 80)


if __name__ == "__main__":
    main()
