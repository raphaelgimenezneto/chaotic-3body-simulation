import os

# ----------------------------
# Physics parameters
# ----------------------------
GRID_SIZE = 200
TIME_STEPS = 600
DT = 0.01
SPACE_SIZE = 1.0
G = 1.0
SOFTENING_FACTOR = 1e-6  # Avoids singularity when distance is zero

# Integration method (used by physics engine)
# Options: "euler", "rk4", "verlet"
INTEGRATOR = "euler"

# ----------------------------
# System Initial Conditions
# ----------------------------
# Masses for Body 1, Body 2, and the Probe (Body 3)
MASSES = [1.0, 1.0, 1.0]

# Initial positions for Body 1 and Body 2, as a fraction of SPACE_SIZE.
# The Probe's initial position is sampled uniformly and defined by X_MIN/X_MAX etc.
BODY1_POS_FRAC = (0.25, 0.25)
BODY2_POS_FRAC = (0.75, 0.75)

# Initial velocities for all three bodies. These are absolute values.
BODY1_VEL = (0.05, 0.0)
BODY2_VEL = (0.0, -0.05)
PROBE_VEL = (0.0, 0.05) # This was previously hardcoded in physics.py

# ----------------------------
# Dataset generation parameters
# ----------------------------
NUM_SIMULATIONS = 5000000
BATCH_SIZE = 50000
OUTPUT_DIR = os.path.join("data", "raw")

# Reproducibility (set to None for non-deterministic runs)
SEED = None

# ----------------------------
# Sampling domain for the Probe (Body 3)
# ----------------------------
USE_ZOOM = False

# Zoom expressed as fractions of SPACE_SIZE to stay consistent
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
assert INTEGRATOR in {"euler", "rk4", "verlet"}, "INTEGRATOR must be one of: euler, rk4, verlet"
assert SPACE_SIZE > 0, "SPACE_SIZE must be > 0"
assert 0 <= X_MIN < X_MAX <= SPACE_SIZE, "Invalid X domain bounds"
assert 0 <= Y_MIN < Y_MAX <= SPACE_SIZE, "Invalid Y domain bounds"
assert NUM_SIMULATIONS > 0, "NUM_SIMULATIONS must be > 0"
assert 0 < BATCH_SIZE <= NUM_SIMULATIONS, "BATCH_SIZE must be in (0, NUM_SIMULATIONS]"
assert GRID_SIZE > 0, "GRID_SIZE must be > 0"
assert TIME_STEPS > 0, "TIME_STEPS must be > 0"
assert DT > 0, "DT must be > 0"
assert SOFTENING_FACTOR >= 0, "SOFTENING_FACTOR must be a non-negative number"
assert len(MASSES) == 3 and all(m > 0 for m in MASSES), "MASSES must be a list of 3 positive numbers"