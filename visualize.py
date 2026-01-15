import pandas as pd
import matplotlib.pyplot as plt

import src.config as cfg
from src.analysis import compute_grid_statistics, classify_regions
from src.plotting import plot_analysis

# Visualization-specific resolution (can be lower than the simulation grid for faster rendering)
VISUAL_GRID_SIZE = 100


def main() -> None:
    print("--- STARTING VISUALIZATION ---")
    print(f"Reading data from: {cfg.OUTPUT_DIR}")

    # 1) Load data
    # With pyarrow, pandas can read a folder of parquet files as a single DataFrame.
    try:
        df = pd.read_parquet(cfg.OUTPUT_DIR, engine="pyarrow")
    except FileNotFoundError:
        print("Error: data folder not found. Run main.py first.")
        return
    except ImportError:
        print("Error: missing dependency 'pyarrow'. Install it with: pip install pyarrow")
        return
    except Exception as e:
        print(f"Error while reading parquet data: {e}")
        return

    if len(df) == 0:
        print("Error: no rows found. Make sure main.py generated parquet files in the output folder.")
        return

    # Optional: validate expected columns (adjust names if your dataset uses different ones)
    expected_cols = {"x", "y"}
    if not expected_cols.issubset(df.columns):
        print(f"Error: expected columns {sorted(expected_cols)} but got {sorted(df.columns)}")
        return

    print(f"Loaded {len(df)} points.")
    print("Computing grid statistics...")

    # 2) Domain bounds (from config.py, so the plot matches the simulated domain)
    x_range = [cfg.X_MIN, cfg.X_MAX]
    y_range = [cfg.Y_MIN, cfg.Y_MAX]
    extent = [cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX]

    # 3) Compute statistics
    mean, variance, mask = compute_grid_statistics(df, VISUAL_GRID_SIZE, x_range, y_range)

    # 4) Classify regions
    class_map = classify_regions(mean, variance, mask)

    # 5) Plot
    print("Rendering plots...")
    plot_analysis(mean, variance, class_map, extent)
    plt.show()


if __name__ == "__main__":
    main()
