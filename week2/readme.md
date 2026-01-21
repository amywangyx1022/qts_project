# Crypto Spread Trading Analysis

## Setup

1. Create a `data/` folder in this directory
2. Copy the exchange data folders (`Binance/`, `OKX/`, `Coinbase/`) into `data/`
3. Install dependencies:
   ```bash
   uv sync
   ```

## Running the Analysis

### Option 1: Command Line (Recommended)

Run the optimization script directly for faster execution:

```bash
uv run python scripts/run_optimization.py --n-trials 50 --save-pickle --parallel
```

### Option 2: Jupyter Notebook

Open `crypto_spread_analysis.ipynb` to run the analysis interactively.

> **Note:** The notebook takes significantly longer to run than the command-line script.


More charts in the outputs/plots folder that i generated, but didn't included in the notebook. check out if interested. 

use claude to help with formatting/optimize the speed of the optimizer.