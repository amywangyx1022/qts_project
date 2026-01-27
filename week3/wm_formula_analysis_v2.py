"""
WM Formula Analysis V2 - Focus on the methodology
Key insight: The assignment shows how ratios CHANGE with daily prices
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Data paths
DATA_DIR = Path("/Users/amywang/Documents/Github_repo/qts_project/week3/data")
FC_FILE = DATA_DIR / "ZACKS_FC_2_76e4bece47ce87cb8f221f639c7f829b.csv"
FR_FILE = DATA_DIR / "ZACKS_FR_2_f40c6a304f87d9f492c1f21839d474e2.csv"
MKTV_FILE = DATA_DIR / "ZACKS_MKTV_2_ecb7f768974bbdd26964caefe2fd0378.csv"
SHRS_FILE = DATA_DIR / "ZACKS_SHRS_2_99db6fa97ac677f3c0d45a9fa9a70196.csv"
PRICES_FILE = DATA_DIR / "QUOTEMEDIA_PRICES_247f636d651d8ef83d8ca1e756cf5ee4.csv"

def load_data():
    """Load all WM data"""
    print("Loading data...")

    # FC data
    fc_df = pd.read_csv(FC_FILE, low_memory=False)
    wm_fc = fc_df[fc_df['ticker'] == 'WM'].copy()
    wm_fc['filing_date'] = pd.to_datetime(wm_fc['filing_date'])

    # FR data
    fr_df = pd.read_csv(FR_FILE)
    wm_fr = fr_df[fr_df['ticker'] == 'WM'].copy()

    # MKTV data
    mktv_df = pd.read_csv(MKTV_FILE)
    wm_mktv = mktv_df[mktv_df['ticker'] == 'WM'].copy()

    # SHRS data
    shrs_df = pd.read_csv(SHRS_FILE)
    wm_shrs = shrs_df[shrs_df['ticker'] == 'WM'].copy()

    # Merge fundamentals
    wm_fund = wm_fc.merge(wm_fr, on=['ticker', 'per_end_date'], how='left')
    wm_fund = wm_fund.merge(wm_mktv, on=['ticker', 'per_end_date'], how='left', suffixes=('', '_mktv'))
    wm_fund = wm_fund.merge(wm_shrs, on=['ticker', 'per_end_date'], how='left', suffixes=('', '_shrs'))

    # Load WM prices
    chunks = []
    for chunk in pd.read_csv(PRICES_FILE, chunksize=1000000):
        wm_chunk = chunk[chunk['ticker'] == 'WM']
        if len(wm_chunk) > 0:
            chunks.append(wm_chunk)
    wm_prices = pd.concat(chunks, ignore_index=True)
    wm_prices['date'] = pd.to_datetime(wm_prices['date'])
    wm_prices = wm_prices.sort_values('date')

    print(f"Loaded {len(wm_fund)} fundamental records, {len(wm_prices)} price records")

    return wm_fund, wm_prices

def calculate_ratios_timeseries(wm_fund, wm_prices):
    """
    Calculate ratio time series following the assignment methodology:
    1. Use most recent fundamental data (based on filing date)
    2. Update market cap daily with prices
    3. Recalculate ratios daily
    """

    print("\n" + "="*80)
    print("CALCULATING RATIO TIME SERIES")
    print("="*80)

    # Focus on the period around 2023-07-27
    target_period = wm_fund[wm_fund['per_end_date'] == '2023-06-30'].iloc[0]

    print("\nTarget period: 2023-06-30 (filed 2023-07-26)")
    print(f"tot_lterm_debt: {target_period['tot_lterm_debt']}")
    print(f"net_lterm_debt: {target_period['net_lterm_debt']}")
    print(f"eps_diluted_net: {target_period['eps_diluted_net']}")
    print(f"basic_net_eps: {target_period['basic_net_eps']}")
    print(f"shares_out: {target_period['shares_out']}")
    print(f"mkt_val (at per_end_date): {target_period['mkt_val']}")
    print(f"FR/tot_debt_tot_equity: {target_period['tot_debt_tot_equity']}")
    print(f"FR/ret_invst: {target_period['ret_invst']}")

    # Extract fundamental values
    tot_debt = target_period['tot_lterm_debt']  # millions
    net_debt = target_period['net_lterm_debt']  # millions
    shares = target_period['shares_out']  # millions
    eps_diluted = target_period['eps_diluted_net']  # dollars per share
    eps_basic = target_period['basic_net_eps']  # dollars per share
    mkt_val_per_end = target_period['mkt_val']  # millions
    filing_date = target_period['filing_date']

    # Get price at per_end_date to establish baseline
    per_end_date = pd.to_datetime('2023-06-30')
    price_at_per_end = wm_prices[wm_prices['date'] <= per_end_date].iloc[-1]['adj_close']
    print(f"\nPrice at per_end_date (2023-06-30): {price_at_per_end}")

    # Calculate the "return" R from the pre-computed ret_invst
    # ret_invst = R / (Debt + MktCap)
    # So: R = ret_invst * (Debt + MktCap)
    ret_invst_reported = target_period['ret_invst']

    # Assignment note says to use net_debt where available, otherwise tot_debt
    debt_for_roi = net_debt if not pd.isna(net_debt) else tot_debt

    # Calculate R (operating income) from the reported ret_invst
    R = ret_invst_reported * (debt_for_roi + mkt_val_per_end)
    print(f"\nInferred R (operating income): {ret_invst_reported} * ({debt_for_roi} + {mkt_val_per_end}) = {R}")

    # Now simulate the ratio updates for days around filing date
    start_date = filing_date
    end_date = filing_date + pd.Timedelta(days=5)

    date_range = wm_prices[(wm_prices['date'] >= start_date) & (wm_prices['date'] <= end_date)]

    print(f"\n{'='*80}")
    print(f"RATIO TIME SERIES AROUND FILING DATE")
    print(f"{'='*80}")
    print(f"{'Date':<12} {'Price':<10} {'MktCap':<12} {'Debt/MktCap':<14} {'ROI':<12} {'P/E':<12}")
    print(f"{'-'*80}")

    results = []

    for _, row in date_range.iterrows():
        date = row['date']
        price = row['adj_close']

        # Update market cap with current price
        mkt_cap_current = price * shares  # millions

        # Calculate ratios
        # 1. Debt/MktCap - which debt and which formula?
        debt_mktcap_v1 = tot_debt / mkt_cap_current  # total debt / current mkt cap
        debt_mktcap_v2 = target_period['tot_debt_tot_equity']  # From FR table (static)

        # The key question: how does FR/tot_debt_tot_equity update with price?
        # It's reported as total_debt / total_equity
        # Assignment says "pretend mkt cap and book equity are equivalent"
        # So we should update it as: tot_debt / mkt_cap_current
        debt_mktcap_v3 = tot_debt / mkt_cap_current

        # But assignment Table 6.0.2 note says it uses "FC/net_lterm_debt"
        # Maybe for the ROI calculation only?

        # 2. ROI - recalculate with updated market cap
        roi_current = R / (debt_for_roi + mkt_cap_current)

        # 3. P/E - straightforward
        pe_diluted = price / eps_diluted
        pe_basic = price / eps_basic

        print(f"{date.strftime('%Y-%m-%d'):<12} {price:>9.2f} {mkt_cap_current:>11.2f} {debt_mktcap_v3:>13.6f} {roi_current:>11.6f} {pe_diluted:>11.6f}")

        results.append({
            'date': date,
            'price': price,
            'mkt_cap': mkt_cap_current,
            'debt_mktcap': debt_mktcap_v3,
            'roi': roi_current,
            'pe': pe_diluted
        })

    # Check against expected values for 2023-07-27
    expected_date = pd.to_datetime('2023-07-27')
    result_2023_07_27 = [r for r in results if r['date'] == expected_date]

    if result_2023_07_27:
        r = result_2023_07_27[0]
        print(f"\n{'='*80}")
        print(f"COMPARISON WITH EXPECTED VALUES (2023-07-27)")
        print(f"{'='*80}")
        print(f"{'Metric':<20} {'Calculated':<15} {'Expected':<15} {'Error':<15}")
        print(f"{'-'*80}")

        expected = {
            'debt_mktcap': 2.346040,
            'roi': 2.975598,
            'pe': 106.538755
        }

        for metric in ['debt_mktcap', 'roi', 'pe']:
            calc = r[metric]
            exp = expected[metric]
            error = abs(calc - exp)
            error_pct = error / exp * 100
            print(f"{metric:<20} {calc:>14.6f} {exp:>14.6f} {error:>14.6f} ({error_pct:>5.2f}%)")

    return results

def test_alternative_formulas(wm_fund, wm_prices):
    """
    Test alternative interpretations of the formulas
    """
    print(f"\n{'='*80}")
    print("TESTING ALTERNATIVE FORMULAS")
    print(f"{'='*80}")

    # Get the specific data point
    target = wm_fund[wm_fund['per_end_date'] == '2023-06-30'].iloc[0]
    price_2023_07_27 = wm_prices[wm_prices['date'] == '2023-07-27'].iloc[0]['adj_close']

    tot_debt = target['tot_lterm_debt']
    net_debt = target['net_lterm_debt']
    shares = target['shares_out']
    mkt_cap_current = price_2023_07_27 * shares
    eps_diluted = target['eps_diluted_net']

    # Expected values
    exp_debt_mktcap = 2.346040
    exp_roi = 2.975598
    exp_pe = 106.538755

    print(f"\nTarget date: 2023-07-27")
    print(f"Price: {price_2023_07_27:.6f}")
    print(f"Shares: {shares:.6f} million")
    print(f"MktCap: {mkt_cap_current:.6f} million")
    print(f"Tot debt: {tot_debt:.6f} million")
    print(f"Net debt: {net_debt:.6f} million")

    print(f"\n--- DEBT/MKTCAP FORMULAS ---")

    # The issue: we're getting 0.227 but expect 2.346
    # Ratio is about 10x off, suggesting inverted or different definition

    # Could it be Debt/Equity where Equity = MktCap - Debt?
    equity_v1 = mkt_cap_current - tot_debt
    equity_v2 = mkt_cap_current - net_debt

    print(f"1. tot_debt / mkt_cap = {tot_debt / mkt_cap_current:.6f}")
    print(f"2. tot_debt / (mkt_cap - tot_debt) = {tot_debt / equity_v1:.6f}")
    print(f"3. tot_debt / (mkt_cap - net_debt) = {tot_debt / equity_v2:.6f}")
    print(f"4. mkt_cap / tot_debt = {mkt_cap_current / tot_debt:.6f}")

    # Wait - if equity = mkt_cap - debt, then debt/equity for high debt companies...
    # Let's check: what if it's actually using BOOK equity not market equity?

    # From FR table: tot_debt_tot_equity = 2.2182
    # This is close to expected 2.346
    # The difference might be due to price changes!

    # If the formula is: tot_debt / book_equity
    # And book_equity is approximately mkt_cap_at_per_end_date - tot_debt
    book_equity_estimate = target['mkt_val'] - tot_debt
    print(f"5. tot_debt / book_equity_est = {tot_debt / book_equity_estimate:.6f}")

    # But we need this to update with price...
    # What if book equity is extracted from the FR ratio, then we scale it?

    # From FR: tot_debt_tot_equity = 2.2182 at per_end_date
    # So: book_equity_at_per_end = tot_debt / 2.2182
    book_equity_from_fr = tot_debt / target['tot_debt_tot_equity']
    print(f"6. Book equity from FR = {tot_debt} / {target['tot_debt_tot_equity']} = {book_equity_from_fr:.6f}")

    # Now scale book equity with price changes?
    price_at_per_end = wm_prices[wm_prices['date'] <= pd.to_datetime('2023-06-30')].iloc[-1]['adj_close']
    price_ratio = price_2023_07_27 / price_at_per_end
    book_equity_scaled = book_equity_from_fr * price_ratio
    print(f"7. Book equity scaled: {book_equity_from_fr} * ({price_2023_07_27}/{price_at_per_end}) = {book_equity_scaled:.6f}")
    print(f"8. tot_debt / book_equity_scaled = {tot_debt / book_equity_scaled:.6f}")

    # Hmm, still not matching. Let me think differently...
    # What if it's actually MktCap / Net_Debt?
    print(f"9. mkt_cap / net_debt = {mkt_cap_current / net_debt:.6f}")

    # That's huge... what about Equity / Net_Debt?
    print(f"10. (mkt_cap - tot_debt) / net_debt = {equity_v1 / net_debt:.6f}")

    print(f"\nExpected debt/mktcap: {exp_debt_mktcap:.6f}")
    print("None of these match well!")

    print(f"\n--- Let's check the P/E more carefully ---")
    pe_calc = price_2023_07_27 / eps_diluted
    print(f"Price / EPS_diluted = {price_2023_07_27} / {eps_diluted} = {pe_calc:.6f}")
    print(f"Expected P/E: {exp_pe:.6f}")
    print(f"Error: {abs(pe_calc - exp_pe):.6f} ({abs(pe_calc - exp_pe)/exp_pe * 100:.2f}%)")
    print("P/E is very close - this formula is correct!")

    # Let me examine the previous and next filings to understand the pattern
    print(f"\n--- EXAMINING FILING PATTERN ---")
    recent_filings = wm_fund[wm_fund['per_end_date'] >= '2023-01-01'].sort_values('filing_date')
    print(recent_filings[['per_end_date', 'filing_date', 'tot_lterm_debt', 'net_lterm_debt',
                          'tot_debt_tot_equity', 'ret_invst', 'mkt_val']].to_string())

def main():
    wm_fund, wm_prices = load_data()
    calculate_ratios_timeseries(wm_fund, wm_prices)
    test_alternative_formulas(wm_fund, wm_prices)

if __name__ == '__main__':
    main()
