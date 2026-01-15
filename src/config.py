import os

# ----------------------------
# Physics parameters
# ----------------------------
GRID_SIZE = 200
TIME_STEPS = 600
DT = 0.01
SPACE_SIZE = 1.0
G = 1.0

# ----------------------------
# Dataset generation parameters
# ----------------------------
NUM_SIMULATIONS = 500_000
BATCH_SIZE = 50_000
OUTPUT_DIR = os.path.join("data", "raw")

# Reproducibility (set to None for non-deterministic runs)
SEED = 42

# ----------------------------
# Sampling domain
# ----------------------------
USE_ZOOM = False

# Zoom expressed as fractions of SPACE_SIZE to stay consistent if SPACE_SIZE changes
ZOOM_X_MIN_FRAC, ZOOM_X_MAX_FRAC = 0.4, 0.6
ZOOM_Y_MIN_FRAC, ZOOM_Y_MAX_FRAC = 0.4, 0.6

if USE_ZOOM:
    X_MIN, X_MAX = ZOOM_X_MIN_FRAC * SPACE_SIZE, ZOOM_X_MAX_FRAC * SPACE_SIZE
    Y_MIN, Y_MAX = ZOOM_Y_MIN_FRAC * SPACE_SIZE, ZOOM_Y_MAX_FRAC * SPACE_SIZE
else:
    X_MIN, X_MAX = 0.0, SPACE_SIZE
    Y_MIN, Y_MAX = 0.0, SPACE_SIZE

# ----------------------------
# Sanity checks (fail fast)
# ----------------------------
assert SPACE_SIZE > 0, "SPACE_SIZE must be > 0"
assert 0 <= X_MIN < X_MAX <= SPACE_SIZE, "Invalid X domain bounds"
assert 0 <= Y_MIN < Y_MAX <= SPACE_SIZE, "Invalid Y domain bounds"
assert NUM_SIMULATIONS > 0, "NUM_SIMULATIONS must be > 0"
assert 0 < BATCH_SIZE <= NUM_SIMULATIONS, "BATCH_SIZE must be in (0, NUM_SIMULATIONS]"
assert GRID_SIZE > 0, "GRID_SIZE must be > 0"
assert TIME_STEPS > 0, "TIME_STEPS must be > 0"
assert DT > 0, "DT must be > 0"
