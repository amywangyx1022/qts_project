"""Data loading and regularization for crypto spread trading."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .config import EXCHANGES, DATA_DIR


def load_parquet_files(exchange: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all parquet files for an exchange and combine them.
    Only loads necessary columns and filters trades during load for efficiency.

    Args:
        exchange: Exchange name (Binance, Coinbase, OKX)
        data_dir: Base data directory (defaults to config DATA_DIR)

    Returns:
        Combined DataFrame with trade data only
    """
    if data_dir is None:
        data_dir = DATA_DIR

    exchange_dir = data_dir / exchange
    parquet_files = sorted(exchange_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {exchange_dir}")

    dfs = []
    for file in parquet_files:
        print(f"  Loading {file.name}...", end=" ", flush=True)
        # Only load needed columns
        df = pd.read_parquet(file, columns=["ts", "rec_type", "price"])
        # Filter trades immediately to reduce memory
        df = df[df["rec_type"] == "T"][["ts", "price"]]
        print(f"{len(df)} trades")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def regularize_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regularize trade data to 1-second windows.
    Takes the last trade price observed at the beginning of each second.

    Args:
        df: DataFrame with 'ts' (timestamp) and 'price' columns

    Returns:
        DataFrame with 1-second regular intervals, forward-filled
    """
    print("  Regularizing to 1-second intervals...", end=" ", flush=True)

    # Convert price to float if needed
    if df["price"].dtype == object:
        df = df.copy()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Ensure timestamp is datetime
    df["ts"] = pd.to_datetime(df["ts"])

    # Sort by timestamp
    df = df.sort_values("ts")

    # Floor to second and take last price per second using resample (more efficient)
    df = df.set_index("ts")
    last_price_per_second = df["price"].resample("1s").last()

    # Forward-fill gaps
    last_price_per_second = last_price_per_second.ffill()

    # Drop any initial NaN values
    last_price_per_second = last_price_per_second.dropna()

    # Convert to DataFrame
    result = last_price_per_second.reset_index()
    result.columns = ["ts", "price"]

    print(f"{len(result)} rows")
    return result


def filter_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to trade records only (rec_type == 'T')."""
    return df[df["rec_type"] == "T"].copy()


def convert_price_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price column from string to float."""
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def load_exchange_data(
    exchange: str, data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load, filter, and regularize data for a single exchange.

    Args:
        exchange: Exchange name
        data_dir: Base data directory

    Returns:
        Regularized 1-second price data
    """
    print(f"Loading {exchange}...")

    # Load raw data (already filtered to trades)
    trades = load_parquet_files(exchange, data_dir)

    # Regularize to 1-second intervals
    regularized = regularize_to_seconds(trades)

    # Rename price column to include exchange name
    regularized = regularized.rename(columns={"price": f"price_{exchange}"})

    return regularized


def load_all_exchanges(data_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """
    Load regularized data for all exchanges.

    Args:
        data_dir: Base data directory

    Returns:
        Dictionary mapping exchange name to regularized DataFrame
    """
    exchange_data = {}
    for exchange in EXCHANGES:
        try:
            exchange_data[exchange] = load_exchange_data(exchange, data_dir)
            print(f"Loaded {exchange}: {len(exchange_data[exchange])} rows")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    return exchange_data


def align_exchange_data(exchange_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align all exchange data to a common timestamp index using inner join.

    Args:
        exchange_data: Dictionary mapping exchange name to DataFrame

    Returns:
        Single DataFrame with aligned prices for all exchanges
    """
    if not exchange_data:
        raise ValueError("No exchange data provided")

    # Start with first exchange
    exchanges = list(exchange_data.keys())
    aligned = exchange_data[exchanges[0]].copy()

    # Merge with remaining exchanges
    for exchange in exchanges[1:]:
        df = exchange_data[exchange]
        aligned = pd.merge(aligned, df, on="ts", how="inner")

    # Sort by timestamp
    aligned = aligned.sort_values("ts").reset_index(drop=True)

    return aligned


def load_and_align_all_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Main function to load, regularize, and align all exchange data.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with aligned 1-second prices for all exchanges
    """
    # Load all exchanges
    exchange_data = load_all_exchanges(data_dir)

    # Align to common timestamps
    aligned = align_exchange_data(exchange_data)

    print(f"\nAligned data shape: {aligned.shape}")
    print(f"Time range: {aligned['ts'].min()} to {aligned['ts'].max()}")

    return aligned


def get_spread_pair_data(
    aligned_data: pd.DataFrame,
    exchange_a: str,
    exchange_b: str
) -> pd.DataFrame:
    """
    Extract price data for a specific spread pair.

    Args:
        aligned_data: Aligned DataFrame with all exchange prices
        exchange_a: First exchange name
        exchange_b: Second exchange name

    Returns:
        DataFrame with ts, price_a, price_b columns
    """
    price_col_a = f"price_{exchange_a}"
    price_col_b = f"price_{exchange_b}"

    if price_col_a not in aligned_data.columns:
        raise ValueError(f"Exchange {exchange_a} not in aligned data")
    if price_col_b not in aligned_data.columns:
        raise ValueError(f"Exchange {exchange_b} not in aligned data")

    result = aligned_data[["ts", price_col_a, price_col_b]].copy()
    result = result.rename(columns={price_col_a: "price_a", price_col_b: "price_b"})

    return result


if __name__ == "__main__":
    # Test data loading
    data = load_and_align_all_data()
    print("\nSample data:")
    print(data.head(10))
    print("\nData types:")
    print(data.dtypes)
