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
def compute_forces(pos_x, pos_y, masses, ejected, G, fx, fy, softening_factor):
    """
    Compute pairwise gravitational forces for the 3-body system.

    Notes:
      - Accounts for variable masses.
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

            # Small epsilon avoids division by zero / singularity
            dist = np.sqrt(dist_sq) + softening_factor
            
            # F = G * m1 * m2 / r^2  =>  F/r = G * m1 * m2 / r^3
            factor = G * masses[a] * masses[b] / (dist * dist * dist)

            fx[a] += factor * dx
            fy[a] += factor * dy


@njit(inline="always", fastmath=True)
def euler_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor):
    """
    Perform one Euler integration step (dt) and update ejection status.
    """
    compute_forces(pos_x, pos_y, masses, ejected, G, fx, fy, softening_factor)

    newly_ejected = 0
    for a in range(3):
        if ejected[a]:
            continue
        
        # a = F/m
        acc_x = fx[a] / masses[a]
        acc_y = fy[a] / masses[a]

        vel_x[a] += acc_x * dt
        vel_y[a] += acc_y * dt
        pos_x[a] += vel_x[a] * dt
        pos_y[a] += vel_y[a] * dt

        if not (0.0 <= pos_x[a] <= space_size and 0.0 <= pos_y[a] <= space_size):
            ejected[a] = True
            newly_ejected += 1

    return newly_ejected


@njit(parallel=True, fastmath=True)
def run_simulation_batch_euler(probe_initial_positions, grid_size, space_size, time_steps, dt, G, masses, initial_body_positions, initial_velocities, softening_factor):
    N = probe_initial_positions.shape[0]
    results = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        # --- Per-simulation state variables ---
        pos_x = np.zeros(3)
        pos_y = np.zeros(3)
        vel_x = np.zeros(3)
        vel_y = np.zeros(3)
        fx = np.zeros(3)
        fy = np.zeros(3)
        ejected = np.zeros(3, dtype=np.bool_)

        # --- Set initial conditions from config ---
        # Body 1 (index 0)
        pos_x[0] = initial_body_positions[0, 0]
        pos_y[0] = initial_body_positions[0, 1]
        vel_x[0] = initial_velocities[0, 0]
        vel_y[0] = initial_velocities[0, 1]

        # Body 2 (index 1)
        pos_x[1] = initial_body_positions[1, 0]
        pos_y[1] = initial_body_positions[1, 1]
        vel_x[1] = initial_velocities[1, 0]
        vel_y[1] = initial_velocities[1, 1]
        
        # Probe (Body 3, index 2) - this is the one we sample
        pos_x[2] = probe_initial_positions[i, 0]
        pos_y[2] = probe_initial_positions[i, 1]
        vel_x[2] = initial_velocities[2, 0]
        vel_y[2] = initial_velocities[2, 1]

        # --- Run the simulation loop ---
        total_sum = 0.0
        num_ejected = 0

        for _ in range(time_steps):
            newly_ejected = euler_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor)
            num_ejected += newly_ejected

            if num_ejected == 3:
                break

            for a in range(3):
                if not ejected[a]:
                    xi, yi = get_cell_index(pos_x[a], pos_y[a], grid_size, space_size)
                    total_sum += (yi * grid_size) + xi + 1

        results[i] = total_sum

    return results


def run_simulation_batch(probe_initial_positions, grid_size, space_size, time_steps, dt, G, softening_factor, integrator_id, initial_conditions):
    """
    Public API: dispatch to the chosen integrator.
    Unpacks initial_conditions dict for Numba compatibility.
    """
    # Unpack dictionary into Numba-friendly NumPy arrays
    masses = initial_conditions["masses"]
    initial_body_positions = initial_conditions["positions"]
    initial_velocities = initial_conditions["velocities"]
    
    if integrator_id == 0:
        return run_simulation_batch_euler(
            probe_initial_positions, 
            grid_size, 
            space_size, 
            time_steps, 
            dt, 
            G,
            masses,
            initial_body_positions,
            initial_velocities,
            softening_factor
        )
    elif integrator_id == 1:
        raise ValueError("RK4 integrator not implemented yet (integrator_id=1)")
    elif integrator_id == 2:
        raise ValueError("Verlet integrator not implemented yet (integrator_id=2)")
    else:
        raise ValueError("Unknown integrator_id")