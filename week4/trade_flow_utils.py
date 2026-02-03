"""
Trade Flow Return Prediction - Core Utilities

Vectorized computation engine for high-frequency trade flow analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load parquet file and prepare for analysis.

    Args:
        filepath: Path to parquet file

    Returns:
        Prepared DataFrame with sorted timestamps and optimized dtypes
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)

    # Convert timestamp to datetime
    df['ts'] = pd.to_datetime(df['ts'])

    # Create numeric timestamp (seconds as float) for fast comparison
    df['ts_numeric'] = df['ts'].astype('int64') / 1e9

    # Sort by Exchange and timestamp
    df = df.sort_values(['Exchange', 'ts_numeric']).reset_index(drop=True)

    # Optimize memory with category dtypes
    df['Exchange'] = df['Exchange'].astype('category')
    df['side'] = df['side'].astype('category')

    print(f"Loaded {len(df):,} trades")
    print(f"Exchanges: {df['Exchange'].unique().tolist()}")
    print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return df


def split_train_test_by_exchange(
    df: pd.DataFrame,
    train_fraction: float = 0.4
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split each exchange into train and test sets by time.

    Args:
        df: Full DataFrame
        train_fraction: Fraction of data for training (default 0.4)

    Returns:
        Dictionary: {exchange: {'train': df_train, 'test': df_test}}
    """
    splits = {}

    for exchange in df['Exchange'].unique():
        exchange_df = df[df['Exchange'] == exchange].copy()

        # Find split timestamp (40th percentile)
        split_ts = exchange_df['ts_numeric'].quantile(train_fraction)

        train_df = exchange_df[exchange_df['ts_numeric'] <= split_ts].copy()
        test_df = exchange_df[exchange_df['ts_numeric'] > split_ts].copy()

        splits[exchange] = {
            'train': train_df,
            'test': test_df
        }

        print(f"{exchange}: {len(train_df):,} train, {len(test_df):,} test")

    return splits


def compute_trade_flow_vectorized(
    df: pd.DataFrame,
    tau: float
) -> pd.Series:
    """
    Compute τ-interval trade flow for each trade.

    F(τ)_i = sum(buy quantities) - sum(sell quantities)
             over [t_i - τ, t_i) excluding trade i itself

    Args:
        df: DataFrame with ts_numeric, side, qty columns (single exchange)
        tau: Time window in seconds

    Returns:
        Series of trade flow values
    """
    # Convert side to signed quantities: +qty for 'B', -qty for 'A'
    signed_qty = np.where(df['side'] == 'B', df['qty'].values, -df['qty'].values)
    ts_numeric = df['ts_numeric'].values

    n = len(df)
    trade_flow = np.zeros(n, dtype=np.float64)

    # Vectorized approach using searchsorted
    for i in range(n):
        t_i = ts_numeric[i]
        window_start = t_i - tau

        # Find indices for window [window_start, t_i)
        # Use 'left' to exclude trades at exactly t_i
        left_idx = np.searchsorted(ts_numeric, window_start, side='left')
        right_idx = np.searchsorted(ts_numeric, t_i, side='left')

        # Sum signed quantities in window
        if right_idx > left_idx:
            trade_flow[i] = signed_qty[left_idx:right_idx].sum()

    return pd.Series(trade_flow, index=df.index, name='trade_flow')


def compute_forward_returns_vectorized(
    df: pd.DataFrame,
    T: float
) -> pd.Series:
    """
    Compute T-second forward returns.

    r(T)_i = (price at t_i+T - price at t_i) / price at t_i

    Args:
        df: DataFrame with ts, trade_price columns (single exchange)
        T: Forward time window in seconds

    Returns:
        Series of forward returns
    """
    # Create forward timestamp
    df_work = df[['ts', 'ts_numeric', 'trade_price']].copy()
    df_work['ts_forward'] = df_work['ts'] + pd.Timedelta(seconds=T)

    # Find nearest trade price at t+T using merge_asof
    forward_df = pd.merge_asof(
        df_work[['ts_forward']],
        df[['ts', 'trade_price']].rename(columns={'trade_price': 'forward_price'}),
        left_on='ts_forward',
        right_on='ts',
        direction='forward'
    )

    # Compute returns
    forward_return = (forward_df['forward_price'].values - df['trade_price'].values) / df['trade_price'].values

    return pd.Series(forward_return, index=df.index, name='forward_return')


def train_flow_return_model(
    train_df: pd.DataFrame,
    flow_col: str = 'trade_flow',
    return_col: str = 'forward_return'
) -> Dict:
    """
    Train linear regression: r(T) = β · F(τ) + ε

    Args:
        train_df: Training DataFrame with flow and return columns
        flow_col: Name of trade flow column
        return_col: Name of forward return column

    Returns:
        Dictionary with model parameters and diagnostics
    """
    # Remove NaN values
    valid_mask = train_df[flow_col].notna() & train_df[return_col].notna()
    X = train_df.loc[valid_mask, flow_col].values.reshape(-1, 1)
    y = train_df.loc[valid_mask, return_col].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Compute diagnostics
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)

    # Standard error of beta
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    X_var = np.var(X, ddof=1)
    std_error = np.sqrt(mse / (len(X) * X_var)) if X_var > 0 else np.nan

    return {
        'beta': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'r_squared': float(r_squared),
        'std_error': float(std_error),
        'n_samples': int(len(X))
    }


def predict_returns(
    df: pd.DataFrame,
    beta: float,
    flow_col: str = 'trade_flow'
) -> pd.Series:
    """
    Predict returns: r_hat = β · F(τ)

    Args:
        df: DataFrame with trade flow column
        beta: Regression coefficient
        flow_col: Name of trade flow column

    Returns:
        Series of predicted returns
    """
    return beta * df[flow_col]


def determine_threshold(
    predicted_returns: pd.Series,
    target_participation: float = 0.05
) -> Dict:
    """
    Find threshold j such that ~5% of trades have |r_hat| > j

    Args:
        predicted_returns: Series of predicted returns
        target_participation: Target fraction of trades (default 0.05)

    Returns:
        Dictionary with threshold and actual participation metrics
    """
    # Remove NaN values
    valid_preds = predicted_returns.dropna()
    abs_preds = valid_preds.abs()

    # Find percentile corresponding to target participation
    percentile = 100 * (1 - target_participation)
    threshold = abs_preds.quantile(percentile / 100)

    # Calculate actual participation
    n_above = (abs_preds > threshold).sum()
    actual_participation = n_above / len(valid_preds)

    return {
        'threshold': float(threshold),
        'actual_participation': float(actual_participation),
        'n_trades': int(n_above),
        'percentile': float(percentile)
    }
