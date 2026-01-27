"""
WM Formula Validation Script
Reverse-engineer exact formulas for Assignment Table 6.0.2
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Data file paths
DATA_DIR = Path("/Users/amywang/Documents/Github_repo/qts_project/week3/data")
FC_FILE = DATA_DIR / "ZACKS_FC_2_76e4bece47ce87cb8f221f639c7f829b.csv"
FR_FILE = DATA_DIR / "ZACKS_FR_2_f40c6a304f87d9f492c1f21839d474e2.csv"
MKTV_FILE = DATA_DIR / "ZACKS_MKTV_2_ecb7f768974bbdd26964caefe2fd0378.csv"
SHRS_FILE = DATA_DIR / "ZACKS_SHRS_2_99db6fa97ac677f3c0d45a9fa9a70196.csv"
PRICES_FILE = DATA_DIR / "QUOTEMEDIA_PRICES_247f636d651d8ef83d8ca1e756cf5ee4.csv"

# Expected values from Assignment Table 6.0.2
EXPECTED_VALUES = {
    '2023-07-27': {
        'debt_mktcap': 2.346040,
        'roi': 2.975598,
        'pe': 106.538755
    },
    '2024-04-26': {
        'debt_mktcap': 2.307417,
        'roi': 3.140420,
        'pe': 119.223466
    }
}

# SEC Report values from Table 6.0.1
SEC_REPORTS = {
    '2023-06-30': {
        'filing_date': '2023-07-26',
        'tot_lterm_debt': 14855.0,
        'net_lterm_debt': 282.0,
        'eps_diluted_net': 1.510,
        'basic_net_eps': 1.520,
        'shares_out': 405.06,
        'mkt_val': 70245.41,
        'tot_debt_tot_equity': 2.2182,
        'ret_invst': 2.8141
    },
    '2024-03-31': {
        'filing_date': '2024-04-25',
        'tot_lterm_debt': 15762.0,
        'net_lterm_debt': -158.0,
        'eps_diluted_net': 1.750,
        'basic_net_eps': 1.760,
        'shares_out': 401.08,
        'mkt_val': 85490.86,
        'tot_debt_tot_equity': 2.2744,
        'ret_invst': 3.0954
    }
}

def load_wm_data():
    """Load WM data from all relevant files"""
    print("Loading WM data from ZACKS files...")

    # Load Financial Condition (FC) - debt, EPS
    print(f"Loading FC file: {FC_FILE}")
    fc_cols = ['ticker', 'per_end_date', 'filing_date', 'tot_lterm_debt', 'net_lterm_debt',
               'eps_diluted_net', 'basic_net_eps', 'tot_revnu', 'net_curr_debt']
    fc_df = pd.read_csv(FC_FILE, usecols=fc_cols)
    wm_fc = fc_df[fc_df['ticker'] == 'WM'].copy()
    print(f"Found {len(wm_fc)} FC records for WM")

    # Load Financial Ratios (FR) - pre-computed ratios
    print(f"Loading FR file: {FR_FILE}")
    fr_cols = ['ticker', 'per_end_date', 'tot_debt_tot_equity', 'ret_invst']
    fr_df = pd.read_csv(FR_FILE, usecols=fr_cols)
    wm_fr = fr_df[fr_df['ticker'] == 'WM'].copy()
    print(f"Found {len(wm_fr)} FR records for WM")

    # Load Market Value (MKTV)
    print(f"Loading MKTV file: {MKTV_FILE}")
    mktv_df = pd.read_csv(MKTV_FILE)
    wm_mktv = mktv_df[mktv_df['ticker'] == 'WM'].copy()
    print(f"Found {len(wm_mktv)} MKTV records for WM")

    # Load Shares Outstanding (SHRS)
    print(f"Loading SHRS file: {SHRS_FILE}")
    shrs_df = pd.read_csv(SHRS_FILE)
    wm_shrs = shrs_df[shrs_df['ticker'] == 'WM'].copy()
    print(f"Found {len(wm_shrs)} SHRS records for WM")

    return wm_fc, wm_fr, wm_mktv, wm_shrs

def load_wm_prices():
    """Load WM prices from QUOTEMEDIA (filtering for WM only)"""
    print(f"\nLoading WM prices from QUOTEMEDIA...")
    print(f"File: {PRICES_FILE}")

    # Read prices in chunks to handle 5.4GB file
    chunks = []
    chunk_size = 1000000
    for chunk in pd.read_csv(PRICES_FILE, chunksize=chunk_size):
        wm_chunk = chunk[chunk['ticker'] == 'WM']
        if len(wm_chunk) > 0:
            chunks.append(wm_chunk)

    if chunks:
        wm_prices = pd.concat(chunks, ignore_index=True)
        print(f"Found {len(wm_prices)} price records for WM")
        return wm_prices
    else:
        print("No WM price records found!")
        return pd.DataFrame()

def analyze_filing_date(per_end_date, filing_date, wm_fc, wm_fr, wm_mktv, wm_shrs, wm_prices):
    """
    Analyze a specific filing date and try to reproduce the ratios
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: per_end_date={per_end_date}, filing_date={filing_date}")
    print(f"{'='*80}")

    # Get the next trading day after filing_date (the day ratios are known)
    filing_dt = pd.to_datetime(filing_date)
    next_day = filing_dt + pd.Timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')

    print(f"\nFiling date: {filing_date}")
    print(f"Next trading day (when ratios known): {next_day_str}")

    # Extract data for this period
    fc_data = wm_fc[wm_fc['per_end_date'] == per_end_date].iloc[0]
    fr_data = wm_fr[wm_fr['per_end_date'] == per_end_date].iloc[0]
    mktv_data = wm_mktv[wm_mktv['per_end_date'] == per_end_date].iloc[0]
    shrs_data = wm_shrs[wm_shrs['per_end_date'] == per_end_date].iloc[0]

    # Get price on next trading day
    price_data = wm_prices[wm_prices['date'] == next_day_str]
    if len(price_data) == 0:
        print(f"WARNING: No price data for {next_day_str}, trying to forward fill...")
        # Try previous days
        for days_back in range(1, 5):
            alt_date = (next_day - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
            price_data = wm_prices[wm_prices['date'] == alt_date]
            if len(price_data) > 0:
                print(f"Using price from {alt_date} (forward fill)")
                break

    print(f"\n--- RAW DATA FROM ZACKS FILES ---")
    print(f"FC - tot_lterm_debt: {fc_data['tot_lterm_debt']}")
    print(f"FC - net_lterm_debt: {fc_data['net_lterm_debt']}")
    print(f"FC - eps_diluted_net: {fc_data['eps_diluted_net']}")
    print(f"FC - basic_net_eps: {fc_data['basic_net_eps']}")
    print(f"FR - tot_debt_tot_equity: {fr_data['tot_debt_tot_equity']}")
    print(f"FR - ret_invst: {fr_data['ret_invst']}")
    print(f"MKTV - mkt_val: {mktv_data['mkt_val']}")
    print(f"SHRS - shares_out: {shrs_data['shares_out']}")

    if len(price_data) > 0:
        price = price_data.iloc[0]['adj_close']
        print(f"PRICES - adj_close on {next_day_str}: {price}")
    else:
        print(f"ERROR: No price data available for {next_day_str} or nearby dates")
        price = None

    # Get expected values
    expected = EXPECTED_VALUES.get(next_day_str, {})

    print(f"\n--- EXPECTED VALUES (from Assignment Table 6.0.2) ---")
    print(f"Debt/MktCap: {expected.get('debt_mktcap', 'N/A')}")
    print(f"ROI: {expected.get('roi', 'N/A')}")
    print(f"P/E: {expected.get('pe', 'N/A')}")

    # Test various formula hypotheses
    print(f"\n--- TESTING FORMULA HYPOTHESES ---")

    # Extract values
    tot_debt = fc_data['tot_lterm_debt']
    net_debt = fc_data['net_lterm_debt']
    mkt_val_zacks = mktv_data['mkt_val']  # in millions
    shares = shrs_data['shares_out']  # in millions
    eps_diluted = fc_data['eps_diluted_net']
    eps_basic = fc_data['basic_net_eps']
    ret_invst_zacks = fr_data['ret_invst']
    tot_debt_equity_zacks = fr_data['tot_debt_tot_equity']

    if price is not None:
        # Calculate market cap from price and shares
        mkt_cap_calculated = price * shares  # shares in millions, so result in millions

        print(f"\n1. DEBT/MKTCAP RATIO:")
        print(f"   Hypothesis 1 - tot_debt/mkt_val: {tot_debt}/{mkt_val_zacks} = {tot_debt/mkt_val_zacks:.6f}")
        print(f"   Hypothesis 2 - mkt_val/tot_debt (inverted): {mkt_val_zacks}/{tot_debt} = {mkt_val_zacks/tot_debt:.6f}")
        print(f"   Hypothesis 3 - tot_debt/mkt_cap_calc: {tot_debt}/{mkt_cap_calculated} = {tot_debt/mkt_cap_calculated:.6f}")
        print(f"   Hypothesis 4 - FR/tot_debt_tot_equity (pre-computed): {tot_debt_equity_zacks:.6f}")
        print(f"   Hypothesis 5 - net_debt/mkt_val: {net_debt}/{mkt_val_zacks} = {net_debt/mkt_val_zacks:.6f}")
        print(f"   Hypothesis 6 - net_debt/mkt_cap_calc: {net_debt}/{mkt_cap_calculated} = {net_debt/mkt_cap_calculated:.6f}")

        # NOTE: The assignment says "pretend market cap and book equity are equivalent"
        # So FR/tot_debt_tot_equity might be total_debt/book_equity
        # But assignment Table 6.0.2 says it uses MKTV/mkt_val and FC/net_lterm_debt

        # The key insight: assignment footnote says "Using MKTV/mkt_val, FC/net_lterm_debt to infer operating income"
        # This suggests we should use NET debt, not TOTAL debt for debt/mktcap
        # And we need to update market value using current price!

        print(f"\n2. RETURN ON INVESTMENT:")
        print(f"   FR/ret_invst (pre-computed): {ret_invst_zacks:.6f}")
        # ROI formula from assignment: R / (D + M)
        # Need to reverse-engineer R (return/operating income) from previous period
        # Then apply to current market value

        print(f"\n3. PRICE/EARNINGS:")
        pe_diluted = (price * shares) / (eps_diluted * shares)  # simplifies to price/eps
        pe_basic = (price * shares) / (eps_basic * shares)
        pe_simple_diluted = price / eps_diluted
        pe_simple_basic = price / eps_basic
        print(f"   Hypothesis 1 - price/eps_diluted: {price}/{eps_diluted} = {pe_simple_diluted:.6f}")
        print(f"   Hypothesis 2 - price/eps_basic: {price}/{eps_basic} = {pe_simple_basic:.6f}")
        print(f"   Hypothesis 3 - (price*shares)/(eps_diluted*shares): {pe_diluted:.6f}")
        print(f"   Hypothesis 4 - (price*shares)/(eps_basic*shares): {pe_basic:.6f}")
        print(f"   Hypothesis 5 - mkt_cap/total_earnings_diluted: {mkt_cap_calculated}/{eps_diluted*shares} = {mkt_cap_calculated/(eps_diluted*shares):.6f}")

        # Calculate errors
        if 'debt_mktcap' in expected:
            print(f"\n--- ERROR ANALYSIS ---")

            # Test which debt/mktcap formula is closest
            formulas_debt = {
                'tot_debt/mkt_val_zacks': tot_debt/mkt_val_zacks,
                'tot_debt/mkt_cap_calc': tot_debt/mkt_cap_calculated,
                'net_debt/mkt_val_zacks': net_debt/mkt_val_zacks,
                'net_debt/mkt_cap_calc': net_debt/mkt_cap_calculated,
                'FR_tot_debt_tot_equity': tot_debt_equity_zacks
            }

            print(f"\nDebt/MktCap - Expected: {expected['debt_mktcap']:.6f}")
            for name, value in formulas_debt.items():
                error = abs(value - expected['debt_mktcap'])
                error_pct = error / expected['debt_mktcap'] * 100
                print(f"  {name}: {value:.6f}, error: {error:.6f} ({error_pct:.2f}%)")

            print(f"\nROI - Expected: {expected['roi']:.6f}")
            print(f"  FR_ret_invst: {ret_invst_zacks:.6f}, error: {abs(ret_invst_zacks - expected['roi']):.6f}")

            print(f"\nP/E - Expected: {expected['pe']:.6f}")
            formulas_pe = {
                'price/eps_diluted': pe_simple_diluted,
                'price/eps_basic': pe_simple_basic,
            }
            for name, value in formulas_pe.items():
                error = abs(value - expected['pe'])
                error_pct = error / expected['pe'] * 100
                print(f"  {name}: {value:.6f}, error: {error:.6f} ({error_pct:.2f}%)")

    return fc_data, fr_data, mktv_data, shrs_data, price_data

def main():
    print("="*80)
    print("WM FORMULA VALIDATION - Reverse Engineering Assignment Table 6.0.2")
    print("="*80)

    # Load data
    wm_fc, wm_fr, wm_mktv, wm_shrs = load_wm_data()
    wm_prices = load_wm_prices()

    # Analyze the two key filing dates
    # 1. per_end_date=2023-06-30, filing_date=2023-07-26, ratio_date=2023-07-27
    analyze_filing_date('2023-06-30', '2023-07-26', wm_fc, wm_fr, wm_mktv, wm_shrs, wm_prices)

    # 2. per_end_date=2024-03-31, filing_date=2024-04-25, ratio_date=2024-04-26
    analyze_filing_date('2024-03-31', '2024-04-25', wm_fc, wm_fr, wm_mktv, wm_shrs, wm_prices)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
