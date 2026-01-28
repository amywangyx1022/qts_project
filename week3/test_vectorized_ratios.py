"""
Test Script for Vectorized Ratio Computation

This script validates the new vectorized implementation against known WM values
before running on the full dataset.

Usage:
    python test_vectorized_ratios.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# Import the vectorized function
from compute_ratios_vectorized import compute_all_ratios_vectorized, validate_against_old_implementation

# Setup paths
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')

def load_zacks_data():
    """Load ZACKS fundamental data."""
    print("Loading ZACKS data...")

    zacks_files = {
        'fc': 'ZACKS_FC_1f6c50082c38bd5040d1450b90d0bb23.csv',
        'fr': 'ZACKS_FR_d7cf850c0e5da28b7a8d1962f96b9f79.csv',
        'mktv': 'ZACKS_MKTV_e7781e0cf5a3ce3c7418bb300b07cc13.csv',
        'shrs': 'ZACKS_SHRS_7f9d02e90c4004c01d097e84c1e012c4.csv'
    }

    zacks_data = {}
    for key, filename in zacks_files.items():
        filepath = DATA_DIR / filename
        print(f"  Loading {filename}...")
        df = pd.read_csv(filepath)

        # Convert date columns
        if key == 'fc':
            df['filing_date'] = pd.to_datetime(df['filing_date'])
            df['per_end_date'] = pd.to_datetime(df['per_end_date'])
        else:
            df['per_end_date'] = pd.to_datetime(df['per_end_date'])

        zacks_data[key] = df
        print(f"    → {len(df):,} rows")

    return zacks_data


def load_price_data(tickers=None):
    """Load QUOTEMEDIA price data."""
    print("\nLoading QUOTEMEDIA_PRICES...")

    price_file = DATA_DIR / 'QUOTEMEDIA_PRICES_247f636d651d8ef83d8ca1e756cf5ee4.csv'

    # Load in chunks to handle large file
    chunks = []
    for chunk in pd.read_csv(price_file, chunksize=1000000):
        if tickers is not None:
            chunk = chunk[chunk['ticker'].isin(tickers)]
        chunks.append(chunk)

    prices = pd.concat(chunks, ignore_index=True)
    prices['date'] = pd.to_datetime(prices['date'])

    print(f"  → {len(prices):,} price records")
    return prices


def test_single_ticker():
    """Test vectorized implementation on WM ticker only."""
    print("=" * 80)
    print("TEST 1: Single Ticker (WM)")
    print("=" * 80)

    # Load data
    zacks_data = load_zacks_data()
    prices = load_price_data(tickers=['WM'])

    # Run vectorized computation
    print("\nRunning vectorized computation on WM...")
    start_time = time.time()

    ratios_df = compute_all_ratios_vectorized(
        tickers=['WM'],
        fc=zacks_data['fc'],
        fr=zacks_data['fr'],
        mktv=zacks_data['mktv'],
        shrs=zacks_data['shrs'],
        prices=prices,
        start_date='2018-01-01',
        end_date='2023-06-30'
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Completed in {elapsed:.2f} seconds")

    # Check for 2023-07-27 data
    target_date = pd.to_datetime('2023-07-27')
    target_row = ratios_df[ratios_df['date'] == target_date]

    if len(target_row) == 0:
        print(f"\n❌ No data found for WM on 2023-07-27")
        print(f"   Available date range: {ratios_df['date'].min()} to {ratios_df['date'].max()}")
        return False

    # Display results
    print("\n" + "=" * 80)
    print("WM RATIOS ON 2023-07-27")
    print("=" * 80)

    row = target_row.iloc[0]

    # Expected values from assignment
    expected = {
        'debt_mktcap': 2.346040,
        'roi': 2.975598,
        'pe': 106.538755
    }

    print(f"\n{'Metric':<15} {'Calculated':<15} {'Expected':<15} {'Error %':<15} {'Status':<10}")
    print("-" * 80)

    all_pass = True
    for metric, exp_val in expected.items():
        calc_val = row[metric]
        error_pct = abs(calc_val - exp_val) / exp_val * 100
        status = "✓ PASS" if error_pct < 2.0 else "✗ FAIL"

        if error_pct >= 2.0:
            all_pass = False

        print(f"{metric:<15} {calc_val:>14.6f} {exp_val:>14.6f} {error_pct:>14.2f}% {status:<10}")

    print("=" * 80)

    if all_pass:
        print("\n✓✓✓ TEST PASSED - WM ratios within 2% of expected values")
    else:
        print("\n✗✗✗ TEST FAILED - WM ratios exceed 2% error threshold")

    print("=" * 80 + "\n")

    return all_pass


def test_performance_sample():
    """Test performance on a sample of 100 tickers."""
    print("=" * 80)
    print("TEST 2: Performance on 100 Tickers")
    print("=" * 80)

    # Load data
    zacks_data = load_zacks_data()

    # Get 100 random tickers
    all_tickers = zacks_data['fc']['ticker'].unique()
    sample_tickers = list(np.random.choice(all_tickers, size=min(100, len(all_tickers)), replace=False))

    print(f"\nSelected {len(sample_tickers)} random tickers")

    # Load prices for sample
    prices = load_price_data(tickers=sample_tickers)

    # Run vectorized computation
    print("\nRunning vectorized computation...")
    start_time = time.time()

    ratios_df = compute_all_ratios_vectorized(
        tickers=sample_tickers,
        fc=zacks_data['fc'],
        fr=zacks_data['fr'],
        mktv=zacks_data['mktv'],
        shrs=zacks_data['shrs'],
        prices=prices,
        start_date='2018-01-01',
        end_date='2023-06-30'
    )

    elapsed = time.time() - start_time

    # Calculate extrapolated time for full dataset
    tickers_per_sec = len(sample_tickers) / elapsed
    full_dataset_size = 1661  # From plan
    extrapolated_time = full_dataset_size / tickers_per_sec

    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"  Tickers processed: {len(sample_tickers)}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Tickers/second: {tickers_per_sec:.1f}")
    print(f"  Extrapolated time for 1,661 tickers: {extrapolated_time:.1f} seconds ({extrapolated_time/60:.1f} minutes)")
    print("=" * 80 + "\n")

    if extrapolated_time < 60:
        print("✓✓✓ PERFORMANCE TARGET MET - Under 1 minute for full dataset")
        return True
    else:
        print("✗✗✗ PERFORMANCE TARGET MISSED - Over 1 minute for full dataset")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VECTORIZED RATIO COMPUTATION - TEST SUITE")
    print("=" * 80 + "\n")

    # Test 1: Single ticker validation
    test1_pass = test_single_ticker()

    # Test 2: Performance test
    test2_pass = test_performance_sample()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Test 1 (WM Validation): {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"  Test 2 (Performance):   {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print("=" * 80)

    if test1_pass and test2_pass:
        print("\n✓✓✓ ALL TESTS PASSED - Ready for full dataset")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED - Review before full deployment")
        return 1


if __name__ == '__main__':
    exit(main())
