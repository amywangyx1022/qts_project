"""
FX Carry Strategy Utilities
----------------------------
Data loading, zero-curve bootstrap, bond pricing, P&L engine, and analytics
for a GBP-funded EM FX carry strategy.
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Group 1: Data Loading & Cleaning
# ---------------------------------------------------------------------------

# Bloomberg CSV column-to-country mapping
SHEET_MAP = {
    1: ("TRY", ["GTTRY1YR Govt", "GTTRY5YR Govt", "GTTRY10Y Govt"]),
    2: ("NGN", ["GTNGN1YR Govt", "GTNGN5YR Govt", "GTNGN10Y Govt"]),
    3: ("PKR", ["GTPKR1Y Govt", "GTPKR5Y Govt", "GTPKR10Y Govt"]),
    4: ("ZAR", ["GTZAR1YR Govt", "GTZAR5YR Govt", "GTZAR10Y Govt"]),
    5: ("BRL", ["GTBRL1Y Govt", "GTBRL5Y Govt", "GTBRL10Y Govt"]),
}

TENOR_LABEL = {"1Y": 1.0, "5Y": 5.0, "10Y": 10.0}


def parse_bloomberg_ragged_csv(filepath: str, country_code: str) -> dict[str, pd.DataFrame]:
    """Parse Bloomberg's ragged (date, yield) pair CSV structure.

    Each CSV has 6 columns: date1, yield1, date2, yield2, date3, yield3
    corresponding to 1Y, 5Y, 10Y tenors (each with its own date series).

    Returns dict like {'1Y': DataFrame(date, yield), '5Y': ..., '10Y': ...}
    """
    raw = pd.read_csv(filepath, header=0)

    result = {}
    tenor_labels = ["1Y", "5Y", "10Y"]

    for i, label in enumerate(tenor_labels):
        date_col = raw.columns[2 * i]
        yield_col = raw.columns[2 * i + 1]

        df = raw[[date_col, yield_col]].copy()
        df.columns = ["date", "yield"]

        # Coerce yields — handle #N/A, blanks, sentinel dates
        df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Drop rows where either is NaT/NaN
        df = df.dropna(subset=["date", "yield"])

        # Filter sentinel dates (before 2000)
        df = df[df["date"] >= "2000-01-01"]

        # Filter out yield values that look like Excel date serials (< 1.0 for yields that should be >1%)
        # Actually yields could be low, so just filter obvious garbage
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")
        df.index.name = "date"

        # Convert from percentage to decimal
        df["yield"] = df["yield"] / 100.0

        result[label] = df

    return result


def load_all_em_curves(data_dir: str = "week5") -> dict[str, dict[str, pd.DataFrame]]:
    """Load all EM yield curve data from Bloomberg CSVs.

    Returns nested dict: {country_code: {tenor: DataFrame}}
    """
    base = Path(data_dir)
    all_curves = {}

    for sheet_num, (country, _headers) in SHEET_MAP.items():
        filepath = base / f"EmergingMkt_YC_Download_BBERG_Sheets_{sheet_num}.csv"
        if filepath.exists():
            all_curves[country] = parse_bloomberg_ragged_csv(str(filepath), country)

    return all_curves


def fetch_boe_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SONIA and 5Y par yield from Bank of England API.

    Reuses exact pattern from Bank_of_England_Data_Fetch.ipynb.

    Parameters
    ----------
    start_date, end_date : str like '01/Jan/2019'

    Returns
    -------
    DataFrame with columns: sonia_rate, gbp_5y_par_yield, indexed by date
    """
    url = "http://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/54.0.2840.90 Safari/537.36"
        )
    }
    series_codes = "IUDSOIA,IUDSNPY"
    payload = {
        "SeriesCodes": series_codes,
        "Datefrom": start_date,
        "Dateto": end_date,
        "CSVF": "TN",
        "UsingCodes": "Y",
        "VPD": "Y",
        "VFD": "N",
    }

    response = requests.get(url, params=payload, headers=headers)
    assert response.status_code == 200, f"BoE API returned {response.status_code}"

    df = pd.read_csv(io.BytesIO(response.content))
    df["date"] = pd.to_datetime(df["DATE"])
    df = df.rename(columns={"IUDSOIA": "sonia_rate", "IUDSNPY": "gbp_5y_par_yield"})
    df = df.drop(columns=["DATE"])
    df = df.set_index("date").sort_index()

    # Convert from percentage to decimal
    df["sonia_rate"] = df["sonia_rate"] / 100.0
    df["gbp_5y_par_yield"] = df["gbp_5y_par_yield"] / 100.0

    return df


def fetch_fx_rates(
    pairs: list[str],
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch spot FX rates from Nasdaq Data Link (EDI/CUR datatables API).

    Parameters
    ----------
    pairs : list of str like ['USDGBP', 'USDTRY', ...]
        The 'USD' prefix is stripped to get the currency code for the API.
    start_date, end_date : str like '2019-01-01'
    api_key : Nasdaq Data Link API key

    Returns
    -------
    DataFrame with one column per pair, indexed by date
    Convention: units of foreign currency per 1 USD
    """
    if api_key is None:
        # Try multiple paths for .env (works from both repo root and week5/)
        for env_path in [Path(__file__).parent / ".env", Path("week5") / ".env", Path(".env")]:
            if env_path.exists():
                load_dotenv(env_path)
                break
        api_key = os.getenv("NASDAQ_DATALINK_API")

    all_fx = {}
    base_url = "https://data.nasdaq.com/api/v3/datatables/EDI/CUR.csv"

    for pair in pairs:
        # Extract currency code: 'USDGBP' -> 'GBP'
        ccy_code = pair.replace("USD", "")

        # Fetch with pagination support
        all_rows = []
        params = {
            "api_key": api_key,
            "code": ccy_code,
            "date.gte": start_date,
            "date.lte": end_date,
        }

        while True:
            resp = requests.get(base_url, params=params)
            if resp.status_code != 200:
                warnings.warn(f"Failed to fetch {pair} ({ccy_code}): HTTP {resp.status_code}")
                break

            text = resp.text
            # Datatables API appends cursor info after the CSV data
            # Split on the cursor metadata line
            lines = text.strip().split("\n")
            csv_lines = []
            next_cursor = None
            for line in lines:
                if line.startswith("cursor_id"):
                    # This is the pagination metadata header, skip
                    continue
                # Check if line looks like a cursor value (no commas, alphanumeric)
                if len(csv_lines) > 0 and "," not in line and len(line) > 10:
                    next_cursor = line.strip()
                    continue
                csv_lines.append(line)

            chunk_text = "\n".join(csv_lines)
            if len(csv_lines) > 1:  # header + at least one data row
                chunk_df = pd.read_csv(io.StringIO(chunk_text))
                all_rows.append(chunk_df)

            # Check for next page cursor in response
            # The datatables API includes cursor in qopts.next_cursor_id
            if next_cursor:
                params["qopts.cursor_id"] = next_cursor
            else:
                break

        if len(all_rows) == 0:
            warnings.warn(f"No data returned for {pair} ({ccy_code})")
            continue

        df = pd.concat(all_rows, ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Rate column: units of foreign currency per 1 USD
        if "rate" in df.columns:
            all_fx[f"fx_{pair}"] = df["rate"]
        elif "value" in df.columns:
            all_fx[f"fx_{pair}"] = df["value"]

    result = pd.DataFrame(all_fx)
    result.index.name = "date"
    return result


def align_data_to_wednesdays(
    boe_df: pd.DataFrame,
    fx_df: pd.DataFrame,
    em_curves: dict[str, dict[str, pd.DataFrame]],
    start: str = "2019-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Build Wednesday-aligned DataFrame with forward-filled data.

    Uses merge_asof(direction='backward') to fill each Wednesday with the
    most recent available observation.

    Returns wide DataFrame with columns:
        sonia, gbp_5y, fx_USDGBP, fx_USDTRY, TRY_1Y, TRY_5Y, TRY_10Y, ...
    """
    # Build Wednesday index
    wednesdays = pd.date_range(start=start, end=end, freq="W-WED")
    wed_df = pd.DataFrame(index=wednesdays)
    wed_df.index.name = "date"

    # Forward-fill BoE data to Wednesdays
    boe_reset = boe_df.reset_index()
    boe_reset["date"] = pd.to_datetime(boe_reset["date"])
    boe_reset = boe_reset.sort_values("date")

    wed_reset = wed_df.reset_index()
    merged = pd.merge_asof(
        wed_reset, boe_reset, on="date", direction="backward"
    )
    result = merged.set_index("date")

    # Forward-fill FX data
    if not fx_df.empty:
        fx_reset = fx_df.reset_index()
        fx_reset["date"] = pd.to_datetime(fx_reset["date"])
        fx_reset = fx_reset.sort_values("date")

        wed_reset2 = wed_df.reset_index()
        fx_merged = pd.merge_asof(
            wed_reset2, fx_reset, on="date", direction="backward"
        )
        fx_merged = fx_merged.set_index("date")
        for col in fx_merged.columns:
            result[col] = fx_merged[col]

    # Forward-fill EM yield curves
    for country, tenors in em_curves.items():
        for tenor_label, tenor_df in tenors.items():
            col_name = f"{country}_{tenor_label}"
            t_reset = tenor_df.reset_index()
            t_reset["date"] = pd.to_datetime(t_reset["date"])
            t_reset = t_reset.sort_values("date")
            t_reset = t_reset.rename(columns={"yield": col_name})

            wed_reset3 = wed_df.reset_index()
            t_merged = pd.merge_asof(
                wed_reset3, t_reset, on="date", direction="backward"
            )
            t_merged = t_merged.set_index("date")
            result[col_name] = t_merged[col_name]

    return result


# ---------------------------------------------------------------------------
# Group 2: Zero-Coupon Curve Construction
# ---------------------------------------------------------------------------

def compute_zcb_curve_from_swaps(swap_rates: dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap zero-coupon bond rates from swap (par) rates.

    Adapted from professor's compute_zcb_curve() in Zero_And_Spot_Curves.ipynb.

    Parameters
    ----------
    swap_rates : dict mapping tenor (float) -> rate (decimal, e.g. 0.085)
        Must include at least one tenor. Common: {1.0: r1, 5.0: r5}
        or {5.0: r5, 10.0: r10} when 1Y is unavailable.

    Returns
    -------
    (tenors_array, zcb_rates_array) for use with np.interp

    Notes
    -----
    - Semi-annual coupon convention (step=0.5), matching professor's approach
    - For missing short tenors (e.g., no 1Y for ZAR), the shortest available
      tenor's ZCB rate is extrapolated flat below it.
    """
    sorted_tenors = sorted(swap_rates.keys())

    # Start with tenor=0 at rate equal to shortest available swap rate
    # (flat extrapolation assumption for short end)
    known_tenors = [0.0]
    known_rates = [swap_rates[sorted_tenors[0]]]

    for tenor in sorted_tenors:
        spot_rate = swap_rates[tenor]

        if tenor < 0.001:
            continue

        # Previous coupon times (semi-annual, before current tenor)
        times_prev = np.arange(0.5, tenor, 0.5)
        coupon_half_yr = 0.5 * spot_rate

        if len(times_prev) > 0:
            # Interpolate ZCB rates at previous coupon times
            z_prev = np.interp(times_prev, known_tenors, known_rates)
            preceding_coupons_val = (coupon_half_yr * np.exp(-z_prev * times_prev)).sum()
        else:
            preceding_coupons_val = 0.0

        # Solve for ZCB rate at this tenor
        arg = (1.0 - preceding_coupons_val) / (1.0 + coupon_half_yr)
        if arg <= 0:
            # Degenerate case: very high rates make PV of coupons exceed 1.
            # Fall back to the spot rate itself as the ZCB rate.
            zcb_rate = spot_rate
        else:
            zcb_rate = -np.log(arg) / tenor

        known_tenors.append(tenor)
        known_rates.append(zcb_rate)

    return np.array(known_tenors), np.array(known_rates)


# ---------------------------------------------------------------------------
# Group 3: Bond Pricing
# ---------------------------------------------------------------------------

def price_bond(
    zcb_tenors: np.ndarray,
    zcb_rates: np.ndarray,
    coupon_rate: float,
    remaining_maturity: float,
    coupon_freq: int = 4,
) -> float:
    """Price a bond on a ZCB curve.

    Adapted from professor's bond_price() in Zero_And_Spot_Curves.ipynb,
    modified for quarterly coupons as specified by the homework.

    Parameters
    ----------
    zcb_tenors, zcb_rates : arrays from compute_zcb_curve_from_swaps
    coupon_rate : annual coupon rate (decimal)
    remaining_maturity : years to maturity
    coupon_freq : coupons per year (default 4 = quarterly)

    Returns
    -------
    Bond price (should be ~1.0 for a par bond)
    """
    step = 1.0 / coupon_freq
    times = np.arange(remaining_maturity, 0, step=-step)[::-1]

    if len(times) == 0:
        return 1.0

    # Interpolate ZCB rates at each coupon time
    r = np.interp(times, zcb_tenors, zcb_rates)

    # PV of principal at maturity
    pv_principal = np.exp(-remaining_maturity * r[-1])

    # PV of coupon stream
    coupon_payment = coupon_rate / coupon_freq
    pv_coupons = coupon_payment * np.exp(-r * times).sum()

    return pv_principal + pv_coupons


# ---------------------------------------------------------------------------
# Group 4: Weekly P&L Engine
# ---------------------------------------------------------------------------

def _build_swap_dict(row: pd.Series, country: str) -> dict[float, float] | None:
    """Extract available swap rates for a country from the aligned data row.

    Returns None if insufficient data for bootstrap.
    """
    rates = {}
    for label, tenor_val in [("1Y", 1.0), ("5Y", 5.0), ("10Y", 10.0)]:
        col = f"{country}_{label}"
        val = row.get(col, np.nan)
        if pd.notna(val) and val > 0:
            rates[tenor_val] = val

    # Need at least 5Y to be useful
    if 5.0 not in rates:
        return None

    return rates


def compute_weekly_pnl(
    notional: float,
    sonia_start: float,
    gbp_5y_start: float,
    s5_em_start: float,
    em_swap_rates_start: dict[float, float],
    em_swap_rates_end: dict[float, float],
    fx_em_start: float,
    fx_em_end: float,
    fx_gbp_start: float,
    fx_gbp_end: float,
    leverage: int = 5,
    spread_bps: float = 50,
    filter_bps: float = 50,
) -> dict:
    """Compute weekly P&L for one EM currency position.

    Parameters
    ----------
    notional : USD notional per currency ($10MM / num_active, or $10MM total)
    sonia_start : SONIA rate at week start (decimal)
    gbp_5y_start : UK 5Y par yield at week start (decimal)
    s5_em_start : EM 5Y swap rate at week start (decimal)
    em_swap_rates_start, _end : {tenor: rate} dicts for ZCB bootstrap
    fx_em_start, _end : EM currency per 1 USD
    fx_gbp_start, _end : GBP per 1 USD
    leverage : leverage factor (default 5)
    spread_bps : borrowing spread over SONIA in bps (default 50)
    filter_bps : minimum spread for entry in bps (default 50)

    Returns
    -------
    dict with keys: active, lending_pnl_usd, borrowing_cost_usd, total_pnl_usd
    """
    result = {
        "active": False,
        "lending_pnl_usd": 0.0,
        "borrowing_cost_usd": 0.0,
        "total_pnl_usd": 0.0,
    }

    # Entry filter: EM 5Y must exceed UK 5Y by at least filter_bps
    if s5_em_start < gbp_5y_start + filter_bps / 10000.0:
        return result

    result["active"] = True

    # --- Lending leg (buy EM 5Y bond) ---
    # Convert notional to EM currency
    bond_notional_em = notional * fx_em_start

    # Bootstrap ZCB curves
    zcb_start_tenors, zcb_start_rates = compute_zcb_curve_from_swaps(em_swap_rates_start)
    zcb_end_tenors, zcb_end_rates = compute_zcb_curve_from_swaps(em_swap_rates_end)

    # Price at par (week start) — coupon = 5Y swap rate
    coupon_rate = s5_em_start
    p_start = price_bond(zcb_start_tenors, zcb_start_rates, coupon_rate, 5.0)

    # Reprice at week end with new ZCB curve, shorter maturity
    remaining = 5.0 - 1.0 / 52.0
    p_end = price_bond(zcb_end_tenors, zcb_end_rates, coupon_rate, remaining)

    # Full position P&L: captures both bond price change AND FX impact on principal.
    # We bought face_value = bond_notional_em / p_start units at week start.
    # At week end, sell at p_end and convert to USD at new FX rate.
    face_value_em = bond_notional_em / p_start
    bond_value_usd_end = face_value_em * p_end / fx_em_end
    lending_pnl_usd = bond_value_usd_end - notional

    # --- Borrowing leg (borrow in GBP) ---
    # Borrow 80% of notional in GBP (GBP ~0.8 per USD)
    borrow_gbp = 0.8 * notional * fx_gbp_start
    interest_gbp = borrow_gbp * (sonia_start + spread_bps / 10000.0) / 52.0
    repay_usd = (borrow_gbp + interest_gbp) / fx_gbp_end
    borrowing_cost_usd = -(repay_usd - 0.8 * notional)

    result["lending_pnl_usd"] = lending_pnl_usd
    result["borrowing_cost_usd"] = borrowing_cost_usd
    result["total_pnl_usd"] = lending_pnl_usd + borrowing_cost_usd

    return result


def run_carry_strategy(
    aligned_data: pd.DataFrame,
    currencies: list[str],
    notional_usd: float = 10_000_000,
    leverage: int = 5,
    spread_bps: float = 50,
    filter_bps: float = 50,
) -> pd.DataFrame:
    """Run the full carry strategy across all currencies and weeks.

    Parameters
    ----------
    aligned_data : Wednesday-aligned DataFrame from align_data_to_wednesdays
    currencies : list like ['TRY', 'ZAR', 'PKR', 'BRL', 'NGN']
    notional_usd : total USD notional
    leverage : leverage multiplier
    spread_bps : borrowing spread over SONIA
    filter_bps : minimum carry spread for entry

    Returns
    -------
    DataFrame with columns: date, currency, active, lending_pnl_usd,
                             borrowing_cost_usd, total_pnl_usd
    """
    dates = aligned_data.index
    records = []

    for i in range(len(dates) - 1):
        date_start = dates[i]
        date_end = dates[i + 1]
        row_start = aligned_data.loc[date_start]
        row_end = aligned_data.loc[date_end]

        sonia = row_start.get("sonia_rate", np.nan)
        gbp_5y = row_start.get("gbp_5y_par_yield", np.nan)

        if pd.isna(sonia) or pd.isna(gbp_5y):
            continue

        # FX rates for GBP
        fx_gbp_start = row_start.get("fx_USDGBP", np.nan)
        fx_gbp_end = row_end.get("fx_USDGBP", np.nan)

        if pd.isna(fx_gbp_start) or pd.isna(fx_gbp_end):
            continue

        for ccy in currencies:
            # EM 5Y rate
            s5_col = f"{ccy}_5Y"
            s5_start = row_start.get(s5_col, np.nan)

            if pd.isna(s5_start):
                records.append({
                    "date": date_end,
                    "currency": ccy,
                    "active": False,
                    "lending_pnl_usd": 0.0,
                    "borrowing_cost_usd": 0.0,
                    "total_pnl_usd": 0.0,
                })
                continue

            # Build swap rate dicts
            swap_start = _build_swap_dict(row_start, ccy)
            swap_end = _build_swap_dict(row_end, ccy)

            if swap_start is None or swap_end is None:
                records.append({
                    "date": date_end,
                    "currency": ccy,
                    "active": False,
                    "lending_pnl_usd": 0.0,
                    "borrowing_cost_usd": 0.0,
                    "total_pnl_usd": 0.0,
                })
                continue

            # FX rates for EM currency
            fx_col = f"fx_USD{ccy}"
            fx_em_start = row_start.get(fx_col, np.nan)
            fx_em_end = row_end.get(fx_col, np.nan)

            if pd.isna(fx_em_start) or pd.isna(fx_em_end):
                records.append({
                    "date": date_end,
                    "currency": ccy,
                    "active": False,
                    "lending_pnl_usd": 0.0,
                    "borrowing_cost_usd": 0.0,
                    "total_pnl_usd": 0.0,
                })
                continue

            pnl = compute_weekly_pnl(
                notional=notional_usd,
                sonia_start=sonia,
                gbp_5y_start=gbp_5y,
                s5_em_start=s5_start,
                em_swap_rates_start=swap_start,
                em_swap_rates_end=swap_end,
                fx_em_start=fx_em_start,
                fx_em_end=fx_em_end,
                fx_gbp_start=fx_gbp_start,
                fx_gbp_end=fx_gbp_end,
                leverage=leverage,
                spread_bps=spread_bps,
                filter_bps=filter_bps,
            )
            pnl["date"] = date_end
            pnl["currency"] = ccy
            records.append(pnl)

    return pd.DataFrame(records)


def aggregate_portfolio_pnl(trade_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-currency results into portfolio-level weekly P&L.

    Returns DataFrame indexed by date with columns:
        total_pnl, lending_pnl, borrowing_cost, cum_pnl, num_active,
        and per-currency P&L columns.
    """
    # Per-currency P&L pivot
    pnl_pivot = trade_results.pivot_table(
        index="date", columns="currency", values="total_pnl_usd", aggfunc="sum"
    ).fillna(0)

    # Portfolio aggregates
    weekly = trade_results.groupby("date").agg(
        total_pnl=("total_pnl_usd", "sum"),
        lending_pnl=("lending_pnl_usd", "sum"),
        borrowing_cost=("borrowing_cost_usd", "sum"),
        num_active=("active", "sum"),
    )
    weekly["cum_pnl"] = weekly["total_pnl"].cumsum()

    # Merge per-currency columns
    for col in pnl_pivot.columns:
        weekly[f"pnl_{col}"] = pnl_pivot[col]
        weekly[f"cum_{col}"] = pnl_pivot[col].cumsum()

    return weekly


# ---------------------------------------------------------------------------
# Group 5: Analytics
# ---------------------------------------------------------------------------

def compute_sharpe(weekly_pnl: pd.Series) -> float:
    """Annualized Sharpe ratio from weekly P&L series."""
    if weekly_pnl.std() == 0:
        return 0.0
    return (weekly_pnl.mean() / weekly_pnl.std()) * np.sqrt(52)


def compute_max_drawdown(cum_pnl: pd.Series) -> dict:
    """Compute maximum drawdown from cumulative P&L.

    Returns dict with peak, trough, drawdown amount, dates.
    """
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max

    trough_idx = drawdown.idxmin()
    trough_val = cum_pnl.loc[trough_idx]

    # Peak is the running max at the trough
    peak_val = running_max.loc[trough_idx]
    peak_idx = cum_pnl.loc[:trough_idx][cum_pnl.loc[:trough_idx] == peak_val].index[0]

    dd_amount = trough_val - peak_val
    dd_pct = dd_amount / peak_val if peak_val != 0 else 0.0

    return {
        "peak_date": peak_idx,
        "peak_value": peak_val,
        "trough_date": trough_idx,
        "trough_value": trough_val,
        "drawdown_amount": dd_amount,
        "drawdown_pct": dd_pct,
        "drawdown_series": drawdown,
    }


def compute_rolling_sharpe(weekly_pnl: pd.Series, window: int = 26) -> pd.Series:
    """Compute rolling annualized Sharpe ratio (default 26-week / 6-month)."""
    rolling_mean = weekly_pnl.rolling(window).mean()
    rolling_std = weekly_pnl.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(52)


def compute_correlation_matrix(pnl_by_currency: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of per-currency weekly P&L."""
    return pnl_by_currency.corr()


def pca_risk_factors(pnl_by_currency: pd.DataFrame, n_components: int = 3) -> dict:
    """PCA decomposition of per-currency P&L for risk factor analysis.

    Returns dict with explained_variance_ratio, components (loadings), and
    the transformed data.
    """
    from sklearn.decomposition import PCA

    # Drop rows with all zeros or NaN
    clean = pnl_by_currency.dropna()
    # Only keep columns with variance
    active_cols = clean.columns[clean.std() > 0]
    clean = clean[active_cols]

    if clean.shape[0] < n_components or clean.shape[1] < n_components:
        n_components = min(clean.shape[0], clean.shape[1], n_components)

    if n_components == 0:
        return {"explained_variance_ratio": [], "components": pd.DataFrame(), "transformed": pd.DataFrame()}

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(clean)

    components_df = pd.DataFrame(
        pca.components_, columns=active_cols,
        index=[f"PC{i+1}" for i in range(n_components)]
    )
    transformed_df = pd.DataFrame(
        transformed, index=clean.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": components_df,
        "transformed": transformed_df,
    }


# ---------------------------------------------------------------------------
# Convenience: Summary statistics table
# ---------------------------------------------------------------------------

def summary_stats(portfolio: pd.DataFrame, trade_results: pd.DataFrame, currencies: list[str]) -> pd.DataFrame:
    """Build summary statistics table for the strategy."""
    rows = []

    # Portfolio level
    total = portfolio["total_pnl"]
    dd = compute_max_drawdown(portfolio["cum_pnl"])
    rows.append({
        "Currency": "Portfolio",
        "Total P&L (USD)": total.sum(),
        "Mean Weekly P&L": total.mean(),
        "Std Weekly P&L": total.std(),
        "Sharpe (ann.)": compute_sharpe(total),
        "Max Drawdown": dd["drawdown_amount"],
        "Max DD %": dd["drawdown_pct"],
        "Active Weeks": int(portfolio["num_active"].sum()),
    })

    # Per currency
    for ccy in currencies:
        col = f"pnl_{ccy}"
        if col in portfolio.columns:
            s = portfolio[col]
            cum_col = f"cum_{ccy}"
            cum = portfolio[cum_col]
            dd_c = compute_max_drawdown(cum)
            active_weeks = int(trade_results[
                (trade_results["currency"] == ccy) & (trade_results["active"])
            ].shape[0])
            rows.append({
                "Currency": ccy,
                "Total P&L (USD)": s.sum(),
                "Mean Weekly P&L": s.mean(),
                "Std Weekly P&L": s.std(),
                "Sharpe (ann.)": compute_sharpe(s),
                "Max Drawdown": dd_c["drawdown_amount"],
                "Max DD %": dd_c["drawdown_pct"],
                "Active Weeks": active_weeks,
            })

    return pd.DataFrame(rows).set_index("Currency")
