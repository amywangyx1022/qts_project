"""
Quick validation test - demonstrate that formulas work

This script loads WM data and validates the formulas against
the expected values from Assignment Table 6.0.2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ratio_calculator import calculate_financial_ratios, validate_wm_calculation, print_validation_report

# Data paths
DATA_DIR = Path("/Users/amywang/Documents/Github_repo/qts_project/week3/data")
FC_FILE = DATA_DIR / "ZACKS_FC_2_76e4bece47ce87cb8f221f639c7f829b.csv"
FR_FILE = DATA_DIR / "ZACKS_FR_2_f40c6a304f87d9f492c1f21839d474e2.csv"
MKTV_FILE = DATA_DIR / "ZACKS_MKTV_2_ecb7f768974bbdd26964caefe2fd0378.csv"
SHRS_FILE = DATA_DIR / "ZACKS_SHRS_2_99db6fa97ac677f3c0d45a9fa9a70196.csv"
PRICES_FILE = DATA_DIR / "QUOTEMEDIA_PRICES_247f636d651d8ef83d8ca1e756cf5ee4.csv"


def main():
    """Run validation test on WM data"""

    print("=" * 80)
    print("FORMULA VALIDATION TEST")
    print("=" * 80)
    print("\nLoading WM data...")

    # Load FC data
    fc_df = pd.read_csv(FC_FILE, low_memory=False)
    wm_fc = fc_df[fc_df['ticker'] == 'WM']
    wm_fc_2023_q2 = wm_fc[wm_fc['per_end_date'] == '2023-06-30'].iloc[0]
    print(f"✓ Loaded FC data: {len(wm_fc)} records")

    # Load FR data
    fr_df = pd.read_csv(FR_FILE, low_memory=False)
    wm_fr = fr_df[fr_df['ticker'] == 'WM']
    wm_fr_2023_q2 = wm_fr[wm_fr['per_end_date'] == '2023-06-30'].iloc[0]
    print(f"✓ Loaded FR data: {len(wm_fr)} records")

    # Load MKTV data
    mktv_df = pd.read_csv(MKTV_FILE)
    wm_mktv = mktv_df[mktv_df['ticker'] == 'WM']
    wm_mktv_2023_q2 = wm_mktv[wm_mktv['per_end_date'] == '2023-06-30'].iloc[0]
    print(f"✓ Loaded MKTV data: {len(wm_mktv)} records")

    # Load SHRS data
    shrs_df = pd.read_csv(SHRS_FILE)
    wm_shrs = shrs_df[shrs_df['ticker'] == 'WM']
    wm_shrs_2023_q2 = wm_shrs[wm_shrs['per_end_date'] == '2023-06-30'].iloc[0]
    print(f"✓ Loaded SHRS data: {len(wm_shrs)} records")

    # Load WM prices
    print(f"✓ Loading WM prices (this may take a moment)...")
    chunks = []
    for chunk in pd.read_csv(PRICES_FILE, chunksize=1000000):
        wm_chunk = chunk[chunk['ticker'] == 'WM']
        if len(wm_chunk) > 0:
            chunks.append(wm_chunk)
    wm_prices = pd.concat(chunks, ignore_index=True)
    wm_prices['date'] = pd.to_datetime(wm_prices['date'])
    wm_prices = wm_prices.sort_values('date')
    print(f"✓ Loaded price data: {len(wm_prices)} records")

    # Show input data
    print("\n" + "=" * 80)
    print("INPUT DATA (per_end_date: 2023-06-30, filing_date: 2023-07-26)")
    print("=" * 80)
    print("\nFinancial Condition (FC):")
    print(f"  tot_lterm_debt: {wm_fc_2023_q2['tot_lterm_debt']:.2f} MM")
    print(f"  net_lterm_debt: {wm_fc_2023_q2['net_lterm_debt']:.2f} MM")
    print(f"  eps_diluted_net: {wm_fc_2023_q2['eps_diluted_net']:.2f}")
    print(f"  basic_net_eps: {wm_fc_2023_q2['basic_net_eps']:.2f}")

    print("\nFinancial Ratios (FR):")
    print(f"  tot_debt_tot_equity: {wm_fr_2023_q2['tot_debt_tot_equity']:.6f}")
    print(f"  ret_invst: {wm_fr_2023_q2['ret_invst']:.6f}")

    print("\nMarket Value (MKTV):")
    print(f"  mkt_val: {wm_mktv_2023_q2['mkt_val']:.2f} MM")

    print("\nShares Outstanding (SHRS):")
    print(f"  shares_out: {wm_shrs_2023_q2['shares_out']:.2f} MM")

    # Get prices
    price_2023_06_30 = wm_prices[wm_prices['date'] <= '2023-06-30'].iloc[-1]['adj_close']
    price_2023_07_27 = wm_prices[wm_prices['date'] == '2023-07-27'].iloc[0]['adj_close']

    print("\nPrices:")
    print(f"  2023-06-30 (per_end_date): ${price_2023_06_30:.6f}")
    print(f"  2023-07-27 (day after filing): ${price_2023_07_27:.6f}")

    # Run validation
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION")
    print("=" * 80)

    validation_results = validate_wm_calculation(
        fc_data=wm_fc_2023_q2,
        fr_data=wm_fr_2023_q2,
        mktv_data=wm_mktv_2023_q2,
        shrs_data=wm_shrs_2023_q2,
        prices_data=wm_prices,
        calculation_date='2023-07-27'
    )

    print_validation_report(validation_results)

    # Success criteria
    print("\n" + "=" * 80)
    print("VALIDATION SUCCESS CRITERIA")
    print("=" * 80)

    errors = validation_results['errors_pct']
    all_passed = True

    checks = [
        ('Debt/MktCap', errors.get('debt_mktcap', 100), 0.1, "Must be < 0.1%"),
        ('ROI', errors.get('roi', 100), 5.0, "Must be < 5%"),
        ('P/E', errors.get('pe', 100), 5.0, "Must be < 5%")
    ]

    for name, error, threshold, msg in checks:
        passed = error < threshold
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<20} {error:>6.2f}% error  {status:<10} ({msg})")
        all_passed = all_passed and passed

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("✅ FORMULAS ARE CORRECT")
        print("✅ GREEN LIGHT FOR AGENT 1 TO PROCEED")
    else:
        print("❌ SOME CHECKS FAILED")
        print("❌ REVIEW FORMULAS")
    print("=" * 80)


if __name__ == '__main__':
    main()
