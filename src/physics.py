import numpy as np
from numba import njit, prange


@njit(inline="always")
def get_cell_index(x: float, y: float, grid_size: int, space_size: float):
    """
    Map a continuous position (x, y) in [0, space_size] to a grid cell index (xi, yi),
    clamped to valid bounds [0, grid_size-1].
    """
    xi = int(x / space_size * grid_size)
    yi = int(y / space_size * grid_size)

    if xi >= grid_size:
        xi = grid_size - 1
    elif xi < 0:
        xi = 0

    if yi >= grid_size:
        yi = grid_size - 1
    elif yi < 0:
        yi = 0

    return xi, yi


@njit(inline="always", fastmath=True)
def compute_forces(pos_x, pos_y, ejected, G, fx, fy):
    """
    Compute pairwise forces for the 3-body system.

    Notes:
      - Mass is assumed to be 1.0 for all bodies.
      - Writes results into fx, fy (arrays of length 3).
      - Skips ejected bodies.
    """
    fx[:] = 0.0
    fy[:] = 0.0

    for a in range(3):
        if ejected[a]:
            continue
        for b in range(3):
            if a == b or ejected[b]:
                continue

            dx = pos_x[b] - pos_x[a]
            dy = pos_y[b] - pos_y[a]
            dist_sq = dx * dx + dy * dy

            # Small epsilon avoids division by zero / singularity at very small distances
            dist = np.sqrt(dist_sq) + 1e-5
            factor = G / (dist * dist * dist)  # mass = 1.0 assumed

            fx[a] += factor * dx
            fy[a] += factor * dy


@njit(inline="always", fastmath=True)
def euler_step(pos_x, pos_y, vel_x, vel_y, ejected, dt, space_size, G, fx, fy):
    """
    Perform one Euler integration step (dt) and update ejection status.

    Returns
    -------
    newly_ejected : int
        Number of bodies that became ejected during this step.
    """
    compute_forces(pos_x, pos_y, ejected, G, fx, fy)

    newly_ejected = 0
    for a in range(3):
        if ejected[a]:
            continue

        vel_x[a] += fx[a] * dt
        vel_y[a] += fy[a] * dt
        pos_x[a] += vel_x[a] * dt
        pos_y[a] += vel_y[a] * dt

        if not (0.0 <= pos_x[a] <= space_size and 0.0 <= pos_y[a] <= space_size):
            ejected[a] = True
            newly_ejected += 1

    return newly_ejected


@njit(parallel=True, fastmath=True)
def run_simulation_batch(initial_positions, grid_size, space_size, time_steps, dt, G, integrator_id):
    """
    Run the physics simulation for a batch of initial particle positions.

    integrator_id:
      0 = Euler
      1 = RK4 (not implemented yet)
      2 = Verlet (not implemented yet)
    """
    N = initial_positions.shape[0]
    results = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        # State vectors for 3 bodies
        pos_x = np.zeros(3)
        pos_y = np.zeros(3)
        vel_x = np.zeros(3)
        vel_y = np.zeros(3)
        fx = np.zeros(3)
        fy = np.zeros(3)
        ejected = np.zeros(3, dtype=np.bool_)

        # Initial setup: two predefined starting positions + one variable body
        pos_x[0] = 0.25
        pos_x[1] = 0.75
        pos_x[2] = initial_positions[i, 0]

        pos_y[0] = 0.25
        pos_y[1] = 0.75
        pos_y[2] = initial_positions[i, 1]

        # Initial velocities (chosen constants for the demo)
        vel_x[0] = 0.05
        vel_x[1] = 0.0
        vel_x[2] = 0.0

        vel_y[0] = 0.0
        vel_y[1] = -0.05
        vel_y[2] = 0.05

        total_sum = 0.0
        num_ejected = 0

        for _ in range(time_steps):
            # Dispatch integrator (Numba-friendly int switch)
            if integrator_id == 0:
                newly_ejected = euler_step(pos_x, pos_y, vel_x, vel_y, ejected, dt, space_size, G, fx, fy)
            elif integrator_id == 1:
                # Placeholder: RK4 will be added later
                raise ValueError("RK4 integrator not implemented yet (integrator_id=1)")
            elif integrator_id == 2:
                # Placeholder: Verlet will be added later
                raise ValueError("Verlet integrator not implemented yet (integrator_id=2)")
            else:
                raise ValueError("Unknown integrator_id")

            num_ejected += newly_ejected
            if num_ejected == 3:
                break

            # Grid-based accumulation (1-indexed linear cell id)
            for a in range(3):
                if not ejected[a]:
                    xi, yi = get_cell_index(pos_x[a], pos_y[a], grid_size, space_size)
                    total_sum += (yi * grid_size) + xi + 1

        results[i] = total_sum

    return results
