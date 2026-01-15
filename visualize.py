import os
import pandas as pd
import matplotlib.pyplot as plt

import src.config as cfg
from src.analysis import compute_grid_statistics, classify_regions
from src.plotting import plot_analysis

from src.run_metadata import (
    make_run_dir,
    snapshot_config,
    get_git_commit,
    get_git_status_porcelain,
    get_pip_freeze,
    write_json,
    write_text,
)

# Visualization-specific resolution (can be lower than simulation grid for faster rendering)
VISUAL_GRID_SIZE = 150


def main() -> None:
    print("--- STARTING VISUALIZATION ---")
    print(f"Reading data from: {cfg.OUTPUT_DIR}")
    print(f"Configured integrator: {cfg.INTEGRATOR}")

    # 1) Load data
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
        print("Error: no rows found. Make sure main.py generated parquet files.")
        return

    # 2) Domain bounds (match simulation domain)
    x_range = [cfg.X_MIN, cfg.X_MAX]
    y_range = [cfg.Y_MIN, cfg.Y_MAX]
    extent = [cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX]

    # 3) Compute statistics
    mean, var, mask = compute_grid_statistics(df, VISUAL_GRID_SIZE, x_range, y_range)

    # 4) Classify regions
    class_map = classify_regions(mean, var, mask)

    # 5) Plot
    print("Rendering plots...")
    fig = plot_analysis(mean, var, class_map, extent)

    # 6) Save run artifacts (always)
    run_dir = make_run_dir(base_dir="outputs", prefix="vis")

    fig_path = os.path.join(run_dir, "figure.png")
    fig.savefig(fig_path, dpi=200)
    print(f"Saved figure: {fig_path}")

    # Metadata
    commit = get_git_commit()
    status = get_git_status_porcelain()
    pip_freeze = get_pip_freeze()
    cfg_snapshot = snapshot_config(cfg)

    write_text(os.path.join(run_dir, "git_commit.txt"), commit + "\n")
    write_text(os.path.join(run_dir, "git_status_porcelain.txt"), status + "\n")
    write_text(os.path.join(run_dir, "pip_freeze.txt"), pip_freeze + "\n")
    write_json(os.path.join(run_dir, "config_snapshot.json"), cfg_snapshot)

    # Extra run info (handy for quick auditing)
    run_info = {
        "run_type": "visualization",
        "output_dir": run_dir,
        "input_data_dir": cfg.OUTPUT_DIR,
        "visual_grid_size": VISUAL_GRID_SIZE,
        "domain": {
            "x_min": cfg.X_MIN,
            "x_max": cfg.X_MAX,
            "y_min": cfg.Y_MIN,
            "y_max": cfg.Y_MAX,
        },
        "simulation": {
            "integrator": cfg.INTEGRATOR,
            "dt": cfg.DT,
            "time_steps": cfg.TIME_STEPS,
            "grid_size": cfg.GRID_SIZE,
            "space_size": cfg.SPACE_SIZE,
            "G": cfg.G,
            "use_zoom": cfg.USE_ZOOM,
        },
        "git": {
            "commit": commit,
            "is_dirty": bool(status and status != "unknown"),
        },
    }
    write_json(os.path.join(run_dir, "run_info.json"), run_info)

    if status and status != "unknown":
        print("WARNING: working tree has uncommitted changes.")
        print("         Commit hash may not fully describe the code used.")

    plt.show()


if __name__ == "__main__":
    main()
