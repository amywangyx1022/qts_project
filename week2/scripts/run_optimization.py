#!/usr/bin/env python
"""
Main script to run crypto spread trading optimization.

Usage:
    uv run python scripts/run_optimization.py
    uv run python scripts/run_optimization.py --n-trials 100
    uv run python scripts/run_optimization.py --pairs Binance-Coinbase --zeta 0.0
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd

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
    return parser.parse_args()


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

    # Store all results
    all_results = {}
    all_backtest_results = {}

    # Run optimization for each pair and zeta
    print("\n[2/4] Running optimizations...")
    total_runs = len(pairs) * len(zeta_values)
    run_count = 0

    for spread_pair in pairs:
        # Get price data for this pair
        try:
            pair_data = get_spread_pair_data(aligned_data, spread_pair[0], spread_pair[1])
        except ValueError as e:
            print(f"Skipping {spread_pair}: {e}")
            continue

        timestamps = pair_data["ts"]
        price_a = pair_data["price_a"]
        price_b = pair_data["price_b"]

        for zeta in zeta_values:
            run_count += 1
            pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
            print(f"\n--- Run {run_count}/{total_runs}: {pair_name}, zeta={zeta} ---")

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

    # Save results
    print("\n[3/4] Saving results...")

    # Generate and save summary report
    report = generate_summary_report(
        all_results,
        output_path=OUTPUT_DIR / "results" / "optimization_summary.txt",
    )
    print(report)

    # Save detailed results as JSON
    results_json = {}
    for key, opt_result in all_results.items():
        spread_pair, zeta = key
        pair_name = f"{spread_pair[0]}-{spread_pair[1]}"
        key_str = f"{pair_name}_zeta{zeta}"

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

    # Save trials dataframe for each study
    for key, opt_result in all_results.items():
        spread_pair, zeta = key
        pair_name = f"{spread_pair[0]}_{spread_pair[1]}"
        zeta_str = f"zeta{zeta}".replace(".", "p")

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
