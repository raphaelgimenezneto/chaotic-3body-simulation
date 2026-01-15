# ==============================================================================
# src/physics.py
# ------------------------------------------------------------------------------
# This file contains the core physics engine of the simulation.
# All computationally intensive functions are JIT-compiled with Numba for
# high performance. The engine supports multiple integration schemes.
# ==============================================================================

import numpy as np
from numba import njit, prange

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

@njit(inline="always")
def get_cell_index(x: float, y: float, grid_size: int, space_size: float):
    """
    Map a continuous position (x, y) to a discrete grid cell index (xi, yi).

    The position is clamped to the valid grid bounds [0, grid_size-1].
    """
    xi = int(x / space_size * grid_size)
    yi = int(y / space_size * grid_size)

    # Clamp to bounds
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

    This function calculates the force vector on each body and writes the
    results into the pre-allocated `fx` and `fy` arrays. Ejected bodies
    do not exert or experience force.
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

            # Add softening factor to prevent division by zero (singularity)
            dist = np.sqrt(dist_sq) + softening_factor
            
            # Optimization: F_vec = (G * m1 * m2 / r^2) * (d_vec / r)
            # This simplifies to factor * d_vec, where factor = G*m1*m2 / r^3
            factor = G * masses[a] * masses[b] / (dist * dist * dist)

            fx[a] += factor * dx
            fy[a] += factor * dy

# ==============================================================================
# INTEGRATOR STEP FUNCTIONS
# ==============================================================================

@njit(inline="always", fastmath=True)
def euler_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor):
    """
    Perform one Forward Euler integration step (dt).
    """
    compute_forces(pos_x, pos_y, masses, ejected, G, fx, fy, softening_factor)

    newly_ejected = 0
    for a in range(3):
        if ejected[a]:
            continue
        
        acc_x = fx[a] / masses[a]
        acc_y = fy[a] / masses[a]

        vel_x[a] += acc_x * dt
        vel_y[a] += acc_y * dt
        pos_x[a] += vel_x[a] * dt
        pos_y[a] += vel_y[a] * dt

        # Check for ejection after the step
        if not (0.0 <= pos_x[a] <= space_size and 0.0 <= pos_y[a] <= space_size):
            ejected[a] = True
            newly_ejected += 1

    return newly_ejected


@njit(inline="always", fastmath=True)
def verlet_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor):
    """
    Perform one Velocity Verlet integration step (dt).
    Superior to Euler for its energy conservation properties.
    """
    acc_x_t = np.zeros(3)
    acc_y_t = np.zeros(3)

    # 1. Calculate forces/accelerations at the current position (t)
    compute_forces(pos_x, pos_y, masses, ejected, G, fx, fy, softening_factor)
    for a in range(3):
        if not ejected[a]:
            acc_x_t[a] = fx[a] / masses[a]
            acc_y_t[a] = fy[a] / masses[a]

    # 2. Update positions to t+dt using accelerations from time t
    for a in range(3):
        if not ejected[a]:
            pos_x[a] += vel_x[a] * dt + 0.5 * acc_x_t[a] * dt * dt
            pos_y[a] += vel_y[a] * dt + 0.5 * acc_y_t[a] * dt * dt
            
    # 3. Calculate forces/accelerations at the new position (t+dt)
    compute_forces(pos_x, pos_y, masses, ejected, G, fx, fy, softening_factor)

    # 4. Update velocities to t+dt using the average of old and new accelerations
    for a in range(3):
        if not ejected[a]:
            acc_x_t_plus_dt = fx[a] / masses[a]
            acc_y_t_plus_dt = fy[a] / masses[a]
            vel_x[a] += 0.5 * (acc_x_t[a] + acc_x_t_plus_dt) * dt
            vel_y[a] += 0.5 * (acc_y_t[a] + acc_y_t_plus_dt) * dt
    
    # 5. Check for ejections after the full step
    newly_ejected = 0
    for a in range(3):
        if not ejected[a] and not (0.0 <= pos_x[a] <= space_size and 0.0 <= pos_y[a] <= space_size):
            ejected[a] = True
            newly_ejected += 1
    
    return newly_ejected


@njit(inline="always", fastmath=True)
def rk4_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor):
    """
    Perform one 4th-order Runge-Kutta (RK4) integration step (dt).
    Highly accurate but computationally expensive (4 force calculations per step).
    """
    temp_pos_x, temp_pos_y = np.copy(pos_x), np.copy(pos_y)
    
    # Store the derivatives (k values) for position and velocity
    k_pos_x, k_pos_y = np.zeros((4, 3)), np.zeros((4, 3))
    k_vel_x, k_vel_y = np.zeros((4, 3)), np.zeros((4, 3))

    # --- k1 ---
    compute_forces(pos_x, pos_y, masses, ejected, G, fx, fy, softening_factor)
    for a in range(3):
        if not ejected[a]:
            acc_x, acc_y = fx[a] / masses[a], fy[a] / masses[a]
            k_pos_x[0, a] = vel_x[a] * dt
            k_pos_y[0, a] = vel_y[a] * dt
            k_vel_x[0, a] = acc_x * dt
            k_vel_y[0, a] = acc_y * dt

    # --- k2 ---
    for a in range(3): temp_pos_x[a], temp_pos_y[a] = pos_x[a] + 0.5 * k_pos_x[0, a], pos_y[a] + 0.5 * k_pos_y[0, a]
    compute_forces(temp_pos_x, temp_pos_y, masses, ejected, G, fx, fy, softening_factor)
    for a in range(3):
        if not ejected[a]:
            acc_x, acc_y = fx[a] / masses[a], fy[a] / masses[a]
            k_pos_x[1, a] = (vel_x[a] + 0.5 * k_vel_x[0, a]) * dt
            k_pos_y[1, a] = (vel_y[a] + 0.5 * k_vel_y[0, a]) * dt
            k_vel_x[1, a] = acc_x * dt
            k_vel_y[1, a] = acc_y * dt

    # --- k3 ---
    for a in range(3): temp_pos_x[a], temp_pos_y[a] = pos_x[a] + 0.5 * k_pos_x[1, a], pos_y[a] + 0.5 * k_pos_y[1, a]
    compute_forces(temp_pos_x, temp_pos_y, masses, ejected, G, fx, fy, softening_factor)
    for a in range(3):
        if not ejected[a]:
            acc_x, acc_y = fx[a] / masses[a], fy[a] / masses[a]
            k_pos_x[2, a] = (vel_x[a] + 0.5 * k_vel_x[1, a]) * dt
            k_pos_y[2, a] = (vel_y[a] + 0.5 * k_vel_y[1, a]) * dt
            k_vel_x[2, a] = acc_x * dt
            k_vel_y[2, a] = acc_y * dt

    # --- k4 ---
    for a in range(3): temp_pos_x[a], temp_pos_y[a] = pos_x[a] + k_pos_x[2, a], pos_y[a] + k_pos_y[2, a]
    compute_forces(temp_pos_x, temp_pos_y, masses, ejected, G, fx, fy, softening_factor)
    for a in range(3):
        if not ejected[a]:
            acc_x, acc_y = fx[a] / masses[a], fy[a] / masses[a]
            k_pos_x[3, a] = (vel_x[a] + k_vel_x[2, a]) * dt
            k_pos_y[3, a] = (vel_y[a] + k_vel_y[2, a]) * dt
            k_vel_x[3, a] = acc_x * dt
            k_vel_y[3, a] = acc_y * dt

    # --- Combine and update final state ---
    for a in range(3):
        if not ejected[a]:
            pos_x[a] += (k_pos_x[0, a] + 2*k_pos_x[1, a] + 2*k_pos_x[2, a] + k_pos_x[3, a]) / 6.0
            pos_y[a] += (k_pos_y[0, a] + 2*k_pos_y[1, a] + 2*k_pos_y[2, a] + k_pos_y[3, a]) / 6.0
            vel_x[a] += (k_vel_x[0, a] + 2*k_vel_x[1, a] + 2*k_vel_x[2, a] + k_vel_x[3, a]) / 6.0
            vel_y[a] += (k_vel_y[0, a] + 2*k_vel_y[1, a] + 2*k_vel_y[2, a] + k_vel_y[3, a]) / 6.0

    # Check for ejections after the full step
    newly_ejected = 0
    for a in range(3):
        if not ejected[a] and not (0.0 <= pos_x[a] <= space_size and 0.0 <= pos_y[a] <= space_size):
            ejected[a] = True
            newly_ejected += 1

    return newly_ejected

# ==============================================================================
# CORE SIMULATION LOOP
# ==============================================================================

@njit(parallel=True, fastmath=True)
def _run_simulation_batch_internal(
    probe_initial_positions, grid_size, space_size, time_steps, dt, G, 
    masses, initial_body_positions, initial_velocities, softening_factor, 
    integrator_id
):
    """
    Internal, Numba-accelerated simulation loop.

    This function contains the main simulation logic, parallelized over the
    initial conditions. It dispatches to the correct integration step
    function based on the provided `integrator_id`.
    """
    N = probe_initial_positions.shape[0]
    results = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        # --- Per-simulation state variables ---
        pos_x, pos_y = np.zeros(3), np.zeros(3)
        vel_x, vel_y = np.zeros(3), np.zeros(3)
        fx, fy       = np.zeros(3), np.zeros(3)
        ejected      = np.zeros(3, dtype=np.bool_)

        # --- Set initial conditions ---
        pos_x[0], pos_y[0] = initial_body_positions[0, 0], initial_body_positions[0, 1]
        vel_x[0], vel_y[0] = initial_velocities[0, 0], initial_velocities[0, 1]

        pos_x[1], pos_y[1] = initial_body_positions[1, 0], initial_body_positions[1, 1]
        vel_x[1], vel_y[1] = initial_velocities[1, 0], initial_velocities[1, 1]
        
        pos_x[2], pos_y[2] = probe_initial_positions[i, 0], probe_initial_positions[i, 1]
        vel_x[2], vel_y[2] = initial_velocities[2, 0], initial_velocities[2, 1]

        # --- Run the time-step loop ---
        total_sum = 0.0
        num_ejected = 0

        for _ in range(time_steps):
            newly_ejected = 0
            # Dispatch to the correct integrator step function
            if integrator_id == 0:  # Euler
                newly_ejected = euler_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor)
            elif integrator_id == 1:  # RK4
                newly_ejected = rk4_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor)
            elif integrator_id == 2:  # Verlet
                newly_ejected = verlet_step(pos_x, pos_y, vel_x, vel_y, masses, ejected, dt, space_size, G, fx, fy, softening_factor)
            
            num_ejected += newly_ejected
            if num_ejected == 3:
                break

            # Accumulate the metric for all non-ejected bodies
            for a in range(3):
                if not ejected[a]:
                    xi, yi = get_cell_index(pos_x[a], pos_y[a], grid_size, space_size)
                    total_sum += (yi * grid_size) + xi + 1

        results[i] = total_sum

    return results

# ==============================================================================
# PUBLIC API
# ==============================================================================

def run_simulation_batch(
    probe_initial_positions, grid_size, space_size, time_steps, dt, G, 
    softening_factor, integrator_id, initial_conditions
):
    """
    Public API to run a batch of simulations.

    This function serves as a clean interface to the Numba-accelerated core.
    It unpacks the `initial_conditions` dictionary into NumPy arrays that
    Numba can work with and then calls the internal simulation loop.
    """
    masses = initial_conditions["masses"]
    initial_body_positions = initial_conditions["positions"]
    initial_velocities = initial_conditions["velocities"]
    
    return _run_simulation_batch_internal(
        probe_initial_positions=probe_initial_positions,
        grid_size=grid_size,
        space_size=space_size,
        time_steps=time_steps,
        dt=dt,
        G=G,
        masses=masses,
        initial_body_positions=initial_body_positions,
        initial_velocities=initial_velocities,
        softening_factor=softening_factor,
        integrator_id=integrator_id
    )