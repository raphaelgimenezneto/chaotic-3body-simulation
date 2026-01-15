# ==============================================================================
# src/physics.py
# ------------------------------------------------------------------------------
# Optimized physics engine.
# - Utilizes Newton's 3rd law for force reduction.
# - Removes heap allocations from inner loops (Zero-allocation integrators).
# - Unswitches loops to remove branching from hot paths.
# ==============================================================================

import numpy as np
from numba import njit, prange

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

@njit(inline="always")
def get_cell_index_fast(x: float, y: float, grid_size: int, grid_scale: float):
    """
    Optimized mapping of position to grid index.
    Uses pre-computed grid_scale (grid_size / space_size) to avoid division.
    """
    xi = int(x * grid_scale)
    yi = int(y * grid_scale)

    # Branchless clamping helps slightly, but standard if/else is reliable in Numba
    if xi >= grid_size: xi = grid_size - 1
    elif xi < 0: xi = 0

    if yi >= grid_size: yi = grid_size - 1
    elif yi < 0: yi = 0

    return xi, yi


@njit(inline="always", fastmath=True)
def compute_accelerations(pos_x, pos_y, masses, inv_masses, ejected, G, acc_x, acc_y, softening_factor):
    """
    Calculates accelerations directly.
    Optimization: Uses Newton's 3rd law (Action = -Reaction) to halve calculations.
    """
    acc_x[:] = 0.0
    acc_y[:] = 0.0

    # Unrolled loop structure for 3 bodies to ensure vectorization where possible
    # and minimize loop overhead.
    
    # Body 0 vs Body 1
    if not (ejected[0] or ejected[1]):
        dx = pos_x[1] - pos_x[0]
        dy = pos_y[1] - pos_y[0]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq) + softening_factor
        f = (G * masses[0] * masses[1]) / (dist * dist * dist)
        fx = f * dx
        fy = f * dy
        
        acc_x[0] += fx * inv_masses[0]
        acc_y[0] += fy * inv_masses[0]
        acc_x[1] -= fx * inv_masses[1]
        acc_y[1] -= fy * inv_masses[1]

    # Body 0 vs Body 2
    if not (ejected[0] or ejected[2]):
        dx = pos_x[2] - pos_x[0]
        dy = pos_y[2] - pos_y[0]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq) + softening_factor
        f = (G * masses[0] * masses[2]) / (dist * dist * dist)
        fx = f * dx
        fy = f * dy
        
        acc_x[0] += fx * inv_masses[0]
        acc_y[0] += fy * inv_masses[0]
        acc_x[2] -= fx * inv_masses[2]
        acc_y[2] -= fy * inv_masses[2]

    # Body 1 vs Body 2
    if not (ejected[1] or ejected[2]):
        dx = pos_x[2] - pos_x[1]
        dy = pos_y[2] - pos_y[1]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq) + softening_factor
        f = (G * masses[1] * masses[2]) / (dist * dist * dist)
        fx = f * dx
        fy = f * dy
        
        acc_x[1] += fx * inv_masses[1]
        acc_y[1] += fy * inv_masses[1]
        acc_x[2] -= fx * inv_masses[2]
        acc_y[2] -= fy * inv_masses[2]

# ==============================================================================
# INTEGRATOR STEP FUNCTIONS
# ==============================================================================

@njit(inline="always", fastmath=True)
def check_ejections(pos_x, pos_y, ejected, space_size):
    """Checks bounds and updates ejected status."""
    count = 0
    for a in range(3):
        if not ejected[a]:
            # Using simple comparison logic
            if not (0.0 <= pos_x[a] <= space_size and 0.0 <= pos_y[a] <= space_size):
                ejected[a] = True
                count += 1
    return count

@njit(inline="always", fastmath=True)
def rk4_step_optimized(pos_x, pos_y, vel_x, vel_y, masses, inv_masses, ejected, dt, G, softening_factor, 
                       acc_x, acc_y, temp_pos_x, temp_pos_y, k1_v_x, k1_v_y, k2_v_x, k2_v_y, k3_v_x, k3_v_y, k4_v_x, k4_v_y):
    """
    Allocation-free RK4 implementation.
    Uses pre-allocated scratch buffers passed as arguments.
    """
    
    # --- K1 ---
    compute_accelerations(pos_x, pos_y, masses, inv_masses, ejected, G, acc_x, acc_y, softening_factor)
    for a in range(3):
        k1_v_x[a] = acc_x[a]
        k1_v_y[a] = acc_y[a]

    # --- K2 ---
    # Pos + 0.5 * k1_vel * dt | Vel + 0.5 * k1_acc * dt
    dt_half = 0.5 * dt
    for a in range(3):
        temp_pos_x[a] = pos_x[a] + vel_x[a] * dt_half
        temp_pos_y[a] = pos_y[a] + vel_y[a] * dt_half
    
    compute_accelerations(temp_pos_x, temp_pos_y, masses, inv_masses, ejected, G, acc_x, acc_y, softening_factor)
    for a in range(3):
        k2_v_x[a] = acc_x[a]
        k2_v_y[a] = acc_y[a]

    # --- K3 ---
    # Pos + 0.5 * k2_vel * dt (approx via current vel + k2 acc)
    # Standard RK4 for 2nd order ODE:
    # dx = v
    # dv = a(x)
    # k1x = v0           k1v = a(x0)
    # k2x = v0 + k1v*dt/2  k2v = a(x0 + k1x*dt/2)
    
    for a in range(3):
        temp_pos_x[a] = pos_x[a] + (vel_x[a] + k1_v_x[a] * dt_half) * dt_half
        temp_pos_y[a] = pos_y[a] + (vel_y[a] + k1_v_y[a] * dt_half) * dt_half
        
    compute_accelerations(temp_pos_x, temp_pos_y, masses, inv_masses, ejected, G, acc_x, acc_y, softening_factor)
    for a in range(3):
        k3_v_x[a] = acc_x[a]
        k3_v_y[a] = acc_y[a]

    # --- K4 ---
    for a in range(3):
        temp_pos_x[a] = pos_x[a] + (vel_x[a] + k2_v_x[a] * dt_half) * dt
        temp_pos_y[a] = pos_y[a] + (vel_y[a] + k2_v_y[a] * dt_half) * dt

    compute_accelerations(temp_pos_x, temp_pos_y, masses, inv_masses, ejected, G, acc_x, acc_y, softening_factor)
    for a in range(3):
        k4_v_x[a] = acc_x[a]
        k4_v_y[a] = acc_y[a]

    # --- Final Integration ---
    dt_6 = dt / 6.0
    for a in range(3):
        if not ejected[a]:
            # Update Position
            # x_new = x + dt/6 * (k1x + 2k2x + 2k3x + k4x)
            # effectively: x + dt * v + dt^2/6 * (...)
            # Simplified "Symplectic-like" accumulation for position in RK4 is complex, 
            # standard explicit expansion:
            
            # Velocities at steps
            v1x = vel_x[a]
            v2x = vel_x[a] + k1_v_x[a] * dt_half
            v3x = vel_x[a] + k2_v_x[a] * dt_half
            v4x = vel_x[a] + k3_v_x[a] * dt
            
            v1y = vel_y[a]
            v2y = vel_y[a] + k1_v_y[a] * dt_half
            v3y = vel_y[a] + k2_v_y[a] * dt_half
            v4y = vel_y[a] + k3_v_y[a] * dt

            pos_x[a] += dt_6 * (v1x + 2*v2x + 2*v3x + v4x)
            pos_y[a] += dt_6 * (v1y + 2*v2y + 2*v3y + v4y)
            
            vel_x[a] += dt_6 * (k1_v_x[a] + 2*k2_v_x[a] + 2*k3_v_x[a] + k4_v_x[a])
            vel_y[a] += dt_6 * (k1_v_y[a] + 2*k2_v_y[a] + 2*k3_v_y[a] + k4_v_y[a])


# ==============================================================================
# CORE SIMULATION LOOP
# ==============================================================================

@njit(parallel=True, fastmath=True)
def _run_simulation_batch_internal(
    probe_initial_positions, grid_size, space_size, time_steps, dt, G, 
    masses, initial_body_positions, initial_velocities, softening_factor, 
    integrator_id
):
    N = probe_initial_positions.shape[0]
    results = np.zeros(N, dtype=np.float64)
    grid_scale = grid_size / space_size
    
    # Pre-compute inverse masses to use multiplication instead of division
    inv_masses = 1.0 / masses

    for i in prange(N):
        # --- Per-simulation stack/register allocation ---
        # Explicit small arrays are efficiently handled by Numba
        pos_x = np.zeros(3)
        pos_y = np.zeros(3)
        vel_x = np.zeros(3)
        vel_y = np.zeros(3)
        acc_x = np.zeros(3)
        acc_y = np.zeros(3)
        ejected = np.zeros(3, dtype=np.bool_)

        # Setup Initial State
        pos_x[0], pos_y[0] = initial_body_positions[0, 0], initial_body_positions[0, 1]
        vel_x[0], vel_y[0] = initial_velocities[0, 0], initial_velocities[0, 1]

        pos_x[1], pos_y[1] = initial_body_positions[1, 0], initial_body_positions[1, 1]
        vel_x[1], vel_y[1] = initial_velocities[1, 0], initial_velocities[1, 1]
        
        pos_x[2], pos_y[2] = probe_initial_positions[i, 0], probe_initial_positions[i, 1]
        vel_x[2], vel_y[2] = initial_velocities[2, 0], initial_velocities[2, 1]

        total_sum = 0.0
        num_ejected = 0

        # ----------------------------------------------------------------------
        # EULER LOOP
        # ----------------------------------------------------------------------
        if integrator_id == 0:
            for _ in range(time_steps):
                compute_accelerations(pos_x, pos_y, masses, inv_masses, ejected, G, acc_x, acc_y, softening_factor)

                for a in range(3):
                    if not ejected[a]:
                        vel_x[a] += acc_x[a] * dt
                        vel_y[a] += acc_y[a] * dt
                        pos_x[a] += vel_x[a] * dt
                        pos_y[a] += vel_y[a] * dt
                
                newly_ejected = check_ejections(pos_x, pos_y, ejected, space_size)
                num_ejected += newly_ejected
                if num_ejected == 3: break

                # Metric Calculation
                for a in range(3):
                    if not ejected[a]:
                        xi, yi = get_cell_index_fast(pos_x[a], pos_y[a], grid_size, grid_scale)
                        total_sum += (yi * grid_size) + xi + 1

        # ----------------------------------------------------------------------
        # RK4 LOOP
        # ----------------------------------------------------------------------
        elif integrator_id == 1:
            # Scratch buffers for RK4 to avoid allocation in loop
            temp_pos_x = np.zeros(3)
            temp_pos_y = np.zeros(3)
            k1x, k1y = np.zeros(3), np.zeros(3)
            k2x, k2y = np.zeros(3), np.zeros(3)
            k3x, k3y = np.zeros(3), np.zeros(3)
            k4x, k4y = np.zeros(3), np.zeros(3)

            for _ in range(time_steps):
                rk4_step_optimized(pos_x, pos_y, vel_x, vel_y, masses, inv_masses, ejected, dt, G, softening_factor, 
                                   acc_x, acc_y, temp_pos_x, temp_pos_y, k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y)
                
                newly_ejected = check_ejections(pos_x, pos_y, ejected, space_size)
                num_ejected += newly_ejected
                if num_ejected == 3: break

                # Metric Calculation
                for a in range(3):
                    if not ejected[a]:
                        xi, yi = get_cell_index_fast(pos_x[a], pos_y[a], grid_size, grid_scale)
                        total_sum += (yi * grid_size) + xi + 1

        # ----------------------------------------------------------------------
        # VERLET LOOP
        # ----------------------------------------------------------------------
        elif integrator_id == 2:
            # Verlet requires storing acceleration from the start of the step
            prev_acc_x = np.zeros(3)
            prev_acc_y = np.zeros(3)
            
            # Initial force calc for first step
            compute_accelerations(pos_x, pos_y, masses, inv_masses, ejected, G, prev_acc_x, prev_acc_y, softening_factor)

            for _ in range(time_steps):
                # 1. Half-step Velocity & Full-step Position
                dt_half = 0.5 * dt
                for a in range(3):
                    if not ejected[a]:
                        pos_x[a] += vel_x[a] * dt + 0.5 * prev_acc_x[a] * dt * dt
                        pos_y[a] += vel_y[a] * dt + 0.5 * prev_acc_y[a] * dt * dt
                        
                        # Partial velocity update
                        vel_x[a] += prev_acc_x[a] * dt_half
                        vel_y[a] += prev_acc_y[a] * dt_half

                # 2. Compute Forces at new position
                compute_accelerations(pos_x, pos_y, masses, inv_masses, ejected, G, acc_x, acc_y, softening_factor)

                # 3. Final Velocity update with new forces
                for a in range(3):
                    if not ejected[a]:
                        vel_x[a] += acc_x[a] * dt_half
                        vel_y[a] += acc_y[a] * dt_half
                        # Update prev_acc for next iteration
                        prev_acc_x[a] = acc_x[a]
                        prev_acc_y[a] = acc_y[a]

                newly_ejected = check_ejections(pos_x, pos_y, ejected, space_size)
                num_ejected += newly_ejected
                if num_ejected == 3: break

                # Metric Calculation
                for a in range(3):
                    if not ejected[a]:
                        xi, yi = get_cell_index_fast(pos_x[a], pos_y[a], grid_size, grid_scale)
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
    masses = initial_conditions["masses"].astype(np.float64)
    initial_body_positions = initial_conditions["positions"].astype(np.float64)
    initial_velocities = initial_conditions["velocities"].astype(np.float64)
    
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