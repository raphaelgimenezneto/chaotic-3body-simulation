import numpy as np
import pandas as pd


def compute_grid_statistics(df: pd.DataFrame, grid_size: int, x_range, y_range):
    """
    Compute spatial mean and variance over a 2D grid using simulation output data.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'x' (float): x positions
          - 'y' (float): y positions
          - 'total_sum' or 'soma_total' (float): per-sample scalar value to aggregate

    grid_size : int
        Number of bins per axis (grid_size x grid_size).

    x_range, y_range : [min, max]
        Domain bounds used to bin points consistently with the simulation.

    Returns
    -------
    mean_grid : np.ndarray (grid_size, grid_size)
    var_grid  : np.ndarray (grid_size, grid_size)
    mask      : np.ndarray (grid_size, grid_size) of bool
        True where at least one sample fell into the cell.
    """
    # Pick the value column (support both names to avoid breaking old datasets)
    if "total_sum" in df.columns:
        value_col = "total_sum"
    elif "soma_total" in df.columns:
        value_col = "soma_total"
    else:
        raise KeyError("Expected a value column: 'total_sum' or 'soma_total'")

    # np.histogram2d expects range=[[ymin, ymax], [xmin, xmax]] for (y, x) ordering below
    hist_range = [y_range, x_range]

    # 1) Basic histograms: counts, sum(values), sum(values^2)
    counts, _, _ = np.histogram2d(df["y"], df["x"], bins=grid_size, range=hist_range)
    sum_vals, _, _ = np.histogram2d(
        df["y"], df["x"], bins=grid_size, range=hist_range, weights=df[value_col]
    )
    sum_sq, _, _ = np.histogram2d(
        df["y"], df["x"], bins=grid_size, range=hist_range, weights=df[value_col] ** 2
    )

    # 2) Avoid division by zero
    mask = counts > 0
    mean = np.zeros((grid_size, grid_size), dtype=np.float64)
    var = np.zeros((grid_size, grid_size), dtype=np.float64)

    # 3) Statistics
    mean[mask] = sum_vals[mask] / counts[mask]

    # Var = E[X^2] - (E[X])^2
    var[mask] = (sum_sq[mask] / counts[mask]) - (mean[mask] ** 2)

    # Numerical fix for tiny negative values due to floating point
    var[var < 0] = 0.0

    return mean, var, mask


def classify_regions(mean: np.ndarray, var: np.ndarray, mask: np.ndarray):
    """
    Build a quadrant-based classification map (0,1,2,3) using medians
    of mean and variance over valid (mask==True) cells.

    Classes:
      0: High mean / Low var
      1: High mean / High var
      2: Low mean  / High var
      3: Low mean  / Low var

    Cells with mask == False receive -1.
    """
    mean_vals = mean[mask]
    var_vals = var[mask]

    if mean_vals.size > 0:
        mean_threshold = np.median(mean_vals)
        var_threshold = np.median(var_vals)
    else:
        mean_threshold = 0.0
        var_threshold = 0.0

    class_map = np.full(mean.shape, -1, dtype=np.int32)

    is_high_mean = mean >= mean_threshold
    is_high_var = var >= var_threshold

    class_map[mask & is_high_mean & ~is_high_var] = 0
    class_map[mask & is_high_mean & is_high_var] = 1
    class_map[mask & ~is_high_mean & is_high_var] = 2
    class_map[mask & ~is_high_mean & ~is_high_var] = 3

    return class_map
