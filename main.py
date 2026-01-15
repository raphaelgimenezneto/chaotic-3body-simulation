import os
import numpy as np
import pandas as pd

import src.config as cfg
from src.physics import run_simulation_batch


# Map config string -> integer ID (Numba-friendly)
INTEGRATOR_MAP = {
    "euler": 0,
    "rk4": 1,
    "verlet": 2,
}


def main() -> None:
    """
    Entry point for the dataset generation pipeline.

    Pipeline:
      1) Sample initial (x, y) positions uniformly within the configured area
      2) Run the physics simulation in batches (vectorized) to compute an output metric
      3) Save each batch as a Parquet file for efficient downstream analysis

    Output files:
      - One Parquet file per batch: OUTPUT_DIR/batch_00000.parquet, batch_00001.parquet, ...
      - Columns: x (float32), y (float32), total_sum (float32)
    """

    # ----------------------------
    # Basic configuration sanity checks (fail fast)
    # ----------------------------
    assert cfg.X_MAX > cfg.X_MIN, "X_MAX must be greater than X_MIN"
    assert cfg.Y_MAX > cfg.Y_MIN, "Y_MAX must be greater than Y_MIN"
    assert cfg.NUM_SIMULATIONS > 0, "NUM_SIMULATIONS must be greater than 0"
    assert cfg.BATCH_SIZE > 0, "BATCH_SIZE must be greater than 0"
    assert cfg.INTEGRATOR in INTEGRATOR_MAP, f"Unknown integrator: {cfg.INTEGRATOR!r}"

    integrator_id = INTEGRATOR_MAP[cfg.INTEGRATOR]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("--- STARTING SIMULATION ---")
    print(f"Zoom mode: {cfg.USE_ZOOM}")
    print(f"Area: X[{cfg.X_MIN}-{cfg.X_MAX}], Y[{cfg.Y_MIN}-{cfg.Y_MAX}]")
    print(f"Integrator: {cfg.INTEGRATOR}")
    print(f"Total simulations: {cfg.NUM_SIMULATIONS} | Batch size: {cfg.BATCH_SIZE}")

    # Reproducible RNG when cfg.SEED is provided; otherwise uses system entropy.
    seed = getattr(cfg, "SEED", None)
    rng = np.random.default_rng(seed)

    # We write one file per batch to:
    # - keep memory bounded
    # - allow parallel reads/processing later
    num_batches = int(np.ceil(cfg.NUM_SIMULATIONS / cfg.BATCH_SIZE))
    total_saved = 0

    width = cfg.X_MAX - cfg.X_MIN
    height = cfg.Y_MAX - cfg.Y_MIN

    # ----------------------------
    # Prepare system initial conditions from config
    # ----------------------------
    # Convert fractional positions to absolute positions
    body1_pos = (cfg.BODY1_POS_FRAC[0] * cfg.SPACE_SIZE, cfg.BODY1_POS_FRAC[1] * cfg.SPACE_SIZE)
    body2_pos = (cfg.BODY2_POS_FRAC[0] * cfg.SPACE_SIZE, cfg.BODY2_POS_FRAC[1] * cfg.SPACE_SIZE)

    # Package system parameters for Numba. Numba works well with arrays/tuples.
    initial_conditions = {
        "masses": np.array(cfg.MASSES, dtype=np.float64),
        "positions": np.array([body1_pos, body2_pos], dtype=np.float64),
        "velocities": np.array([cfg.BODY1_VEL, cfg.BODY2_VEL, cfg.PROBE_VEL], dtype=np.float64),
    }

    for batch_idx in range(num_batches):
        # If NUM_SIMULATIONS is not a multiple of BATCH_SIZE, the last batch is smaller.
        remaining = cfg.NUM_SIMULATIONS - batch_idx * cfg.BATCH_SIZE
        batch_size = min(cfg.BATCH_SIZE, remaining)

        # ----------------------------
        # 1) Sample initial positions uniformly in the rectangle [X_MIN, X_MAX] x [Y_MIN, Y_MAX]
        # ----------------------------
        rx = rng.random(batch_size)
        ry = rng.random(batch_size)

        x = cfg.X_MIN + (rx * width)
        y = cfg.Y_MIN + (ry * height)
        batch_positions = np.column_stack((x, y))  # shape: (batch_size, 2)

        # ----------------------------
        # 2) Run physics
        # Expected output: one scalar per initial position (shape: (batch_size,))
        # ----------------------------
        results = run_simulation_batch(
            probe_initial_positions=batch_positions,
            grid_size=cfg.GRID_SIZE,
            space_size=cfg.SPACE_SIZE,
            time_steps=cfg.TIME_STEPS,
            dt=cfg.DT,
            G=cfg.G,
            integrator_id=integrator_id,
            initial_conditions=initial_conditions,
        )

        # ----------------------------
        # 3) Save as Parquet
        # float32 reduces storage size while remaining sufficient for most analyses
        # ----------------------------
        df_batch = pd.DataFrame(
            {
                "x": batch_positions[:, 0].astype(np.float32),
                "y": batch_positions[:, 1].astype(np.float32),
                "total_sum": np.asarray(results, dtype=np.float32),
            }
        )

        file_path = os.path.join(cfg.OUTPUT_DIR, f"batch_{batch_idx:05d}.parquet")
        try:
            df_batch.to_parquet(file_path, engine="pyarrow", compression="snappy")
        except ImportError as e:
            raise ImportError(
                "Failed to save Parquet file. Please install 'pyarrow' (pip install pyarrow)."
            ) from e

        total_saved += len(df_batch)
        print(
            f"Batch {batch_idx + 1}/{num_batches} saved. "
            f"({total_saved}/{cfg.NUM_SIMULATIONS})"
        )

    print("--- FINISHED ---")


if __name__ == "__main__":
    main()