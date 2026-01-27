"""
WM Formula Validation Script

Validates the ratio calculator against known WM values from Assignment Table 6.0.2.
Expected values for 2023-07-27:
  - debt_mktcap: 2.346040
  - roi: 2.975598
  - pe: 106.538755
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ratio_calculator import validate_wm_calculation, print_validation_report

DATA_DIR = Path('data')

print('Loading WM data only (efficient filtering)...')

# Load Zacks data - filter WM immediately after loading
fc = pd.read_csv(list(DATA_DIR.glob('ZACKS_FC_2_*.csv'))[0], parse_dates=['per_end_date', 'filing_date'], low_memory=False)
fc = fc[fc['ticker'] == 'WM']
print(f'  FC: {len(fc)} WM rows')

fr = pd.read_csv(list(DATA_DIR.glob('ZACKS_FR_2_*.csv'))[0], parse_dates=['per_end_date'], low_memory=False)
fr = fr[fr['ticker'] == 'WM']
print(f'  FR: {len(fr)} WM rows')

mktv = pd.read_csv(list(DATA_DIR.glob('ZACKS_MKTV_2_*.csv'))[0], parse_dates=['per_end_date'], low_memory=False)
mktv = mktv[mktv['ticker'] == 'WM']
print(f'  MKTV: {len(mktv)} WM rows')

shrs = pd.read_csv(list(DATA_DIR.glob('ZACKS_SHRS_2_*.csv'))[0], parse_dates=['per_end_date'], low_memory=False)
shrs = shrs[shrs['ticker'] == 'WM']
print(f'  SHRS: {len(shrs)} WM rows')

# Get WM data for per_end_date = 2023-06-30
wm_fc = fc[fc['per_end_date'] == '2023-06-30'].iloc[0]
wm_fr = fr[fr['per_end_date'] == '2023-06-30'].iloc[0]
wm_mktv = mktv[mktv['per_end_date'] == '2023-06-30'].iloc[0]
wm_shrs = shrs[shrs['per_end_date'] == '2023-06-30'].iloc[0]

# Load WM prices only - use chunked reading for efficiency
prices_file = list(DATA_DIR.glob('QUOTEMEDIA_PRICES_*.csv'))[0]
print(f'  Loading prices for WM only from {prices_file.name}...')

wm_prices_list = []
for chunk in pd.read_csv(prices_file, parse_dates=['date'], chunksize=100000):
    wm_chunk = chunk[chunk['ticker'] == 'WM']
    if len(wm_chunk) > 0:
        wm_prices_list.append(wm_chunk)
wm_prices = pd.concat(wm_prices_list, ignore_index=True)
print(f'  WM prices loaded: {len(wm_prices)} rows')

print()
print('=' * 60)
print('WM Data for per_end_date = 2023-06-30:')
print('=' * 60)
print(f'  tot_lterm_debt:     {wm_fc["tot_lterm_debt"]}')
print(f'  net_lterm_debt:     {wm_fc["net_lterm_debt"]}')
print(f'  eps_diluted_net:    {wm_fc["eps_diluted_net"]}')
print(f'  basic_net_eps:      {wm_fc["basic_net_eps"]}')
print(f'  tot_debt_tot_equity:{wm_fr["tot_debt_tot_equity"]}')
print(f'  ret_invst:          {wm_fr["ret_invst"]}')
print(f'  mkt_val:            {wm_mktv["mkt_val"]}')
print(f'  shares_out:         {wm_shrs["shares_out"]}')
print()

# Run validation
validation_results = validate_wm_calculation(
    fc_data=wm_fc,
    fr_data=wm_fr,
    mktv_data=wm_mktv,
    shrs_data=wm_shrs,
    prices_data=wm_prices,
    calculation_date='2023-07-27'
)

print_validation_report(validation_results)

# Check if all ratios pass within 5% tolerance
all_pass = all(err < 5 for err in validation_results['errors_pct'].values())
print(f'\nOVERALL VALIDATION: {"PASS" if all_pass else "FAIL"} (all ratios within 5% tolerance)')
