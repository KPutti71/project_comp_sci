import numpy as np
import sympy as sp
import pylbm

from png_to_grid import png_to_grid, plot_grid

# --------------------------------------------------
# 1) Load grid
# --------------------------------------------------
grid = png_to_grid("./data/test4.png")

mask = grid["mask"]
rects = grid["rects"]
dx = grid["dx"]
W, H = grid["W"], grid["H"]
xmin, xmax, ymin, ymax = grid["box"]
left_label, right_label, bottom_label, top_label = grid["labels"]

# optional debug statements and plot
# print("Grid size:", W, H)
# print("Num wall rectangles:", len(rects))
# print("Wall pixels:", (mask == grid["codes"]["wall"]).sum())
# plot_grid(grid)


# --------------------------------------------------
# 2) pylbm symbols and helpers
# --------------------------------------------------
X, Y, LA = sp.symbols("X, Y, LA")
rho, qx, qy = sp.symbols("rho, qx, qy")


def bc_inlet(f, m, x, y, rhoo, uo):
    m[rho] = 0.0
    m[qx] = rhoo * uo
    m[qy] = 0.0


def velocity_magnitude(sol):
    ux = sol.m[qx] / sol.m[rho]
    uy = sol.m[qy] / sol.m[rho]
    return np.sqrt(ux**2 + uy**2).T


# --------------------------------------------------
# 3) Build pylbm obstacles from grid rectangles
# --------------------------------------------------
L_INLET = 0
L_OUTLET = 1
L_WALL = 2

elements = []
for (x0, x1, y0, y1) in rects:
    px = x0 * dx
    py = y0 * dx
    w = (x1 - x0) * dx
    h = (y1 - y0) * dx

    elements.append(
        pylbm.Parallelogram(
            (px, py),
            (w, 0.0),
            (0.0, h),
            label=L_WALL,
            isfluid=False,
        )
    )


# --------------------------------------------------
# 4) Physical / numerical parameters
# --------------------------------------------------
uo = 0.01
mu = 5e-5
la = 1.0
rhoo = 1.0

zeta = 10 * mu
dummy = 3.0 / (la * rhoo * dx)
s1 = 1.0 / (0.5 + zeta * dummy)
s2 = 1.0 / (0.5 + mu * dummy)
s = [0., 0., 0., s1, s1, s1, s1, s2, s2]

dummy = 1.0 / (LA**2 * rhoo)
qx2 = dummy * qx**2
qy2 = dummy * qy**2
q2 = qx2 + qy2
qxy = dummy * qx * qy


# --------------------------------------------------
# 5) pylbm dictionary
# --------------------------------------------------
dico = {
    "box": {
        "x": [xmin, xmax],
        "y": [ymin, ymax],
        "label": [left_label, right_label, bottom_label, top_label],
    },
    "elements": elements,
    "space_step": dx,
    "scheme_velocity": LA,
    "schemes": [
        {
            "velocities": list(range(9)),
            "polynomials": [
                1,
                LA*X, LA*Y,
                3*(X**2+Y**2)-4,
                0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                3*X*(X**2+Y**2)-5*X,
                3*Y*(X**2+Y**2)-5*Y,
                X**2-Y**2,
                X*Y,
            ],
            "relaxation_parameters": s,
            "equilibrium": [
                rho,
                qx, qy,
                -2*rho + 3*q2,
                rho - 3*q2,
                -qx/LA, -qy/LA,
                qx2 - qy2, qxy,
            ],
            "conserved_moments": [rho, qx, qy],
        }
    ],
    "init": {rho: rhoo, qx: 0.0, qy: 0.0},
    "parameters": {LA: la},
    "boundary_conditions": {
        L_INLET:  {"method": {0: pylbm.bc.BouzidiBounceBack},
                   "value": (bc_inlet, (rhoo, uo))},
        L_OUTLET: {"method": {0: pylbm.bc.NeumannX}},
        L_WALL:   {"method": {0: pylbm.bc.BouzidiBounceBack}},
    },
    "generator": "cython",
}


# --------------------------------------------------
# 6) Run + visualize
# --------------------------------------------------
sol = pylbm.Simulation(dico)

viewer = pylbm.viewer.matplotlib_viewer
fig = viewer.Fig()
ax = fig[0]

ax.image(mask.T, cmap="gray", clim=[0, 3])
vel_img = ax.image(velocity_magnitude(sol), cmap="cubehelix", clim=[0, 0.05])


def update(_):
    for _ in range(8):
        sol.one_time_step()

    v = velocity_magnitude(sol)
    if not np.isfinite(v).all():
        print("NaNs/Infs detected at t =", sol.t)
        return

    vel_img.set_data(v)
    ax.title = f"t={sol.t:.4f}"


fig.animate(update, interval=1)
fig.show()
