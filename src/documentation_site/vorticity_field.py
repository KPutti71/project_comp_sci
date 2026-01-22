"""
2D Lattice Boltzmann (pylbm) simulation:
Channel flow past a circular cylinder -> Von Kármán vortex street.
Postprocess: plot (signed) vorticity magnitude.
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import sympy as sp
import pylbm


# =============================================================================
# Symbolic variables (used by pylbm to build the scheme)
# =============================================================================

X, Y, LA = sp.symbols("X, Y, LA")          # lattice variables + scheme velocity parameter
rho, qx, qy = sp.symbols("rho, qx, qy")    # conserved moments: density + momentum components


# =============================================================================
# Boundary condition helpers
# =============================================================================

def inlet_momentum_bc(f, m, x, y):
    """
    Callback used by pylbm boundary condition.

    We impose a target x-momentum at the inlet:
        qx = rho0 * u_in
    so that (roughly) ux = qx/rho ≈ u_in.
    """
    m[qx] = rho0 * u_in


# =============================================================================
# Post-processing: vorticity
# =============================================================================

def vorticity_field(sol):
    """
    Compute a vorticity-like scalar field from the velocity:
        omega = d(uy)/dx - d(ux)/dy

    Here we use centered differences and take abs(...) then return negative
    to match the original plotting convention.
    """
    ux = sol.m[qx] / sol.m[rho]
    uy = sol.m[qy] / sol.m[rho]

    dx = sol.domain.dx

    # centered differences on interior cells
    d_uy_dx = (uy[2:, 1:-1] - uy[0:-2, 1:-1]) / (2.0 * dx)
    d_ux_dy = (ux[1:-1, 2:] - ux[1:-1, 0:-2]) / (2.0 * dx)

    omega = d_uy_dx - d_ux_dy
    return -np.abs(omega)


# =============================================================================
# Physical + numerical parameters
# =============================================================================

# Geometry
radius = 0.05
xmin, xmax = 0.0, 3.0
ymin, ymax = 0.0, 1.0
cylinder_center = [0.3, 0.5 * (ymin + ymax) + (1.0 / 64)]  # matches original (+dx shift)

# Flow / LBM knobs
Re = 20
dx = 1.0 / 64          # spatial step
la = 1.0               # scheme velocity
Tf = 75                # final time

rho0 = 1.0
u_in = la / 20         # inlet/max velocity scale used in the original script

# Viscosities
mu_bulk = 1e-3
eta_shear = rho0 * u_in * (2.0 * radius) / Re  # set by Reynolds number

print(f"Reynolds number: {Re:10.3e}")
print(f"Bulk viscosity : {mu_bulk:10.3e}")
print(f"Shear viscosity: {eta_shear:10.3e}")


# =============================================================================
# LBM relaxation parameters (MRT for D2Q9)
# =============================================================================

# This factor comes from how pylbm nondimensionalizes the relaxation relation.
dummy = 3.0 / (la * rho0 * dx)

s_mu = 1.0 / (0.5 + mu_bulk * dummy)   # bulk-related relaxation
s_eta = 1.0 / (0.5 + eta_shear * dummy)  # shear-related relaxation

# In this model, some modes share relaxation rates
s_q = s_eta
s_es = s_mu

# D2Q9 MRT: 9 relaxation rates (0 for conserved moments)
s = [0.0, 0.0, 0.0, s_mu, s_es, s_q, s_q, s_eta, s_eta]
print(f"relaxation parameters: {s}")


# =============================================================================
# Symbolic helper expressions for equilibrium moments
# =============================================================================

dummy_eq = 1.0 / (LA**2 * rho0)
qx2 = dummy_eq * qx**2
qy2 = dummy_eq * qy**2
q2  = qx2 + qy2
qxy = dummy_eq * qx * qy


# =============================================================================
# Build pylbm configuration dictionary
# =============================================================================

def build_simulation_config():
    # --- Domain / geometry ---------------------------------------------------
    box = {
        "x": [xmin, xmax],
        "y": [ymin, ymax],
        "label": [0, 2, 0, 0],  # box side labels (pylbm convention)
    }

    elements = [
        pylbm.Circle(cylinder_center, radius, label=1)
    ]

    # --- Scheme definition (D2Q9 MRT) ---------------------------------------
    scheme = {
        "velocities": list(range(9)),
        "conserved_moments": [rho, qx, qy],

        # moment basis polynomials (9 of them for D2Q9)
        "polynomials": [
            1, LA * X, LA * Y,
            3 * (X**2 + Y**2) - 4,
            (9 * (X**2 + Y**2)**2 - 21 * (X**2 + Y**2) + 8) / 2,
            3 * X * (X**2 + Y**2) - 5 * X,
            3 * Y * (X**2 + Y**2) - 5 * Y,
            X**2 - Y**2,
            X * Y,
        ],

        "relaxation_parameters": s,

        # equilibrium moments (functions of rho, qx, qy)
        "equilibrium": [
            rho, qx, qy,
            -2 * rho + 3 * q2,
            rho - 3 * q2,
            -qx / LA,
            -qy / LA,
            qx2 - qy2,
            qxy,
        ],
    }

    # --- Initial condition ---------------------------------------------------
    init = {rho: rho0, qx: 0.0, qy: 0.0}

    # --- Boundary conditions -------------------------------------------------
    # 0: inlet-like boundary (bounce-back method + a value callback setting qx)
    # 1: cylinder (bounce-back for no-slip on curved boundary)
    # 2: outflow-like (Neumann in x)
    bcs = {
        0: {"method": {0: pylbm.bc.BouzidiBounceBack}, "value": inlet_momentum_bc},
        1: {"method": {0: pylbm.bc.BouzidiBounceBack}},
        2: {"method": {0: pylbm.bc.NeumannX}},
    }

    # --- Full config ---------------------------------------------------------
    return {
        "box": box,
        "elements": elements,
        "space_step": dx,
        "scheme_velocity": la,
        "parameters": {LA: la},
        "schemes": [scheme],
        "init": init,
        "boundary_conditions": bcs,
        "generator": "cython",
    }


# =============================================================================
# Run simulation
# =============================================================================

def run():
    config = build_simulation_config()
    sol = pylbm.Simulation(config)

    while sol.t < Tf:
        sol.one_time_step()

    return sol


# =============================================================================
# Plot results
# =============================================================================

def plot(sol):
    viewer = pylbm.viewer.matplotlib_viewer
    fig = viewer.Fig()
    ax = fig[0]

    omega = vorticity_field(sol).transpose()
    ax.image(omega, clim=[-3.0, 0.0], cmap="magma")

    # draw cylinder outline (grid coordinates)
    ax.ellipse([cylinder_center[0] / dx, (0.5 * (ymin + ymax)) / dx],
               [radius / dx, radius / dx],
               "r")

    ax.title = f"Vorticity field at t = {sol.t:f}"
    fig.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sol = run()
    plot(sol)