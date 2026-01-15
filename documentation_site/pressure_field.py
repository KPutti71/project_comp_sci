"""
2D Lattice Boltzmann (pylbm) simulation:
Channel flow past a circular cylinder.
Postprocess: plot pressure colormap (LBM equation of state).
"""

import numpy as np
import sympy as sp
import pylbm

# =============================================================================
# Symbolic variables
# =============================================================================

X, Y, LA = sp.symbols("X Y LA")
rho, qx, qy = sp.symbols("rho qx qy")

# =============================================================================
# Set parameters
# =============================================================================

xmin, xmax = 0.0, 3.0
ymin, ymax = 0.0, 1.0
radius = 0.2

Re = 20
dx = 1.0 / 128
la = 1.
Tf = 75
rho0 = 1. # 
mu_bulk = 1e-3

# =============================================================================
# Derived parameters
# =============================================================================

cylinder_center = [0.3, 0.5 * (ymin + ymax) + dx]
u_in = la / 20
eta_shear = rho0 * u_in * (2.0 * radius) / Re

# MRT relaxation parameters (D2Q9)
dummy = 3.0 / (la * rho0 * dx)
s_mu = 1.0 / (0.5 + mu_bulk * dummy)
s_eta = 1.0 / (0.5 + eta_shear * dummy)
s = [0.0, 0.0, 0.0, s_mu, s_mu, s_eta, s_eta, s_eta, s_eta]

# Symbolic helper expressions
inv = 1.0 / (LA**2 * rho0)
qx2 = inv * qx**2
qy2 = inv * qy**2
q2 = qx2 + qy2
qxy = inv * qx * qy

# print(f"Reynolds number: {Re:10.3e}")
# print(f"Bulk viscosity : {mu_bulk:10.3e}")
# print(f"Shear viscosity: {eta_shear:10.3e}")
# print(f"relaxation parameters: {s}")

# =============================================================================
# Function Definitions
# =============================================================================

def inlet_momentum_bc(f, m, x, y):
    m[qx] = rho0 * u_in

def pressure_field(sol):
    """
    cs is the speed of sound and we calculate the pressure P by fluid pressure
    equation: P = rho * cs^2
    https://scispace.com/pdf/lattice-boltzmann-simulation-of-incompressible-fluid-flow-in-5fuap61di2.pdf
    """
    cs2 = 1.0 / 3.0
    return cs2 * sol.m[rho]

def flow_resistance(sol):
    """
    Calculates the flow resistance R in the simulation
    """
    p = pressure_field(sol)
    p_in = p[int(xmin + 2), int((ymin + ymax)/2)] # pressure in
    p_out = p[int(xmax - 2), int((ymin + ymax)/2)] # pressure out
    Q = u_in * (ymax - ymin) # Flowrate
    R = abs(p_in - p_out) / Q # Flow resistance
    return R

# =============================================================================
# Build pylbm configuration
# =============================================================================

def build_simulation_config():
    box = {"x": [xmin, xmax], "y": [ymin, ymax], "label": [0, 2, 0, 0]}
    elements = [pylbm.Circle(cylinder_center, radius, label=1)]

    scheme = {
        "velocities": list(range(9)),
        "conserved_moments": [rho, qx, qy],
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
        "equilibrium": [
            rho, qx, qy,
            -2 * rho + 3 * q2,
            rho - 3 * q2,
            -qx / LA, -qy / LA,
            qx2 - qy2, qxy,
        ],
    }

    init = {rho: rho0, qx: 0.0, qy: 0.0}

    bcs = {
        0: {"method": {0: pylbm.bc.BouzidiBounceBack}, "value": inlet_momentum_bc},
        1: {"method": {0: pylbm.bc.BouzidiBounceBack}},
        2: {"method": {0: pylbm.bc.NeumannX}},
    }

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
    sol = pylbm.Simulation(build_simulation_config())
    while sol.t < Tf:
        sol.one_time_step()
    return sol

# =============================================================================
# Plot pressure
# =============================================================================

def plot(sol):
    viewer = pylbm.viewer.matplotlib_viewer
    fig = viewer.Fig()
    ax = fig[0]

    p = pressure_field(sol)
    img = (p - p.mean()).T

    ax.image(img, cmap="viridis")

    ax.ellipse([cylinder_center[0] / dx, (0.5 * (ymin + ymax)) / dx],
               [radius / dx, radius / dx], "r")
    
    ax.title = f"Pressure field at t = {sol.t:f}"
    fig.show()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sol = run()
    R = flow_resistance(sol)
    print(f"Flow resistance R = {R}")
    plot(sol)
