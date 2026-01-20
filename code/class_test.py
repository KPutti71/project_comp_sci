import numpy as np
import sympy as sp
import pylbm
import matplotlib.pyplot as plt

from png_to_grid import png_to_grid, plot_grid


class Simulation:
    # Labels used by the grid + pylbm box labels
    L_INLET = 0
    L_OUTLET = 1
    L_WALL = 2

    def __init__(
        self,
        png_path: str = "./data/test.png",
        Re: float = 20.0,
        la: float = 1.0,
        Tf: float = 300.0,
        rho0: float = .1,
        mu_bulk: float = 1e-3,
        dt: int = None
    ):
        # save simulation on instance
        self.sol = None

        # --------------------------------------------------
        # 1) Load grid (store everything you’ll need on self)
        # --------------------------------------------------
        self.png_path = png_path
        self.grid = png_to_grid(self.png_path)

        self.mask = self.grid["mask"]
        self.rects = self.grid["rects"]
        self.dx = float(self.grid["dx"])
        self.W, self.H = int(self.grid["W"]), int(self.grid["H"])

        # Domain box comes from the grid
        self.xmin, self.xmax, self.ymin, self.ymax = map(float, self.grid["box"])

        # Box boundary labels come from the grid
        (
            self.left_label,
            self.right_label,
            self.bottom_label,
            self.top_label,
        ) = self.grid["labels"]

        # --------------------------------------------------
        # 2) Physical / simulation parameters
        # --------------------------------------------------
        self.Re = float(Re)
        self.la = float(la)
        self.Tf = float(Tf)
        self.rho0 = float(rho0)
        self.mu_bulk = float(mu_bulk)

        # Inlet velocity (kept from your earlier setup)
        self.u_in = self.la / 20.0

        # --------------------------------------------------
        # 3) Build pylbm obstacles from grid rectangles
        # --------------------------------------------------
        self.elements = self._build_elements_from_rects(self.rects)

        # --------------------------------------------------
        # 4) Symbolic variables
        # --------------------------------------------------
        self.X, self.Y, self.LA = sp.symbols("X Y LA")
        self.rho, self.qx, self.qy = sp.symbols("rho qx qy")

        # --------------------------------------------------
        # 5) Derived / MRT parameters
        # --------------------------------------------------
        # Need some characteristic length for shear viscosity relation.
        # If your PNG encodes a channel, a safe default is the domain height.
        self.char_length = (self.ymax - self.ymin)

        self.eta_shear = self.rho0 * self.u_in * self.char_length / self.Re

        dummy = 3.0 / (self.la * self.rho0 * self.dx)
        s_mu = 1.0 / (0.5 + self.mu_bulk * dummy)
        s_eta = 1.0 / (0.5 + self.eta_shear * dummy)
        self.s = [0.0, 0.0, 0.0, s_mu, s_mu, s_eta, s_eta, s_eta, s_eta]

        inv = 1.0 / (self.LA**2 * self.rho0)
        self.qx2 = inv * self.qx**2
        self.qy2 = inv * self.qy**2
        self.q2 = self.qx2 + self.qy2
        self.qxy = inv * self.qx * self.qy

    # --------------------------------------------------
    # Grid -> pylbm elements
    # --------------------------------------------------
    def _build_elements_from_rects(self, rects):
        elements = []
        for (x0, x1, y0, y1) in rects:
            px = float(x0) * self.dx
            py = float(y0) * self.dx
            w = float(x1 - x0) * self.dx
            h = float(y1 - y0) * self.dx

            elements.append(
                pylbm.Parallelogram(
                    (px, py),
                    (w, 0.0),
                    (0.0, h),
                    label=self.L_WALL,
                    isfluid=False,
                )
            )
        return elements

    # --------------------------------------------------
    # Post-processing helpers
    # --------------------------------------------------
    def pressure_field(self, sol):
        cs2 = 1.0 / 3.0
        return cs2 * sol.m[self.rho]

    def flow_resistance(self, x_offset_cells: int = 2):
        """
        Uses pressure at two x-locations (near inlet/outlet) at mid-height.
        NOTE: Assumes sol.m[rho] indexing is [ix, iy] in lattice-cell coordinates.
        """
        p = self.pressure_field(self.sol)
        mid_y = int(0.5 * (self.ymin + self.ymax) / self.dx)

        # pick two x positions a few cells away from each side of the domain
        x_in = int(self.xmin / self.dx) + int(x_offset_cells)
        x_out = int(self.xmax / self.dx) - int(x_offset_cells)

        p_in = p[x_in, mid_y]
        p_out = p[x_out, mid_y]

        Q = self.u_in * (self.ymax - self.ymin)
        return abs(p_in - p_out) / Q

    # --------------------------------------------------
    # pylbm config
    # --------------------------------------------------
    def build_simulation_config(self):
        # inlet callback must be a plain function (closure), not a bound method
        qx_sym = self.qx
        rho0 = self.rho0
        u_in = self.u_in

        def inlet_momentum_bc(f, m, x, y):
            m[qx_sym] = rho0 * u_in

        # IMPORTANT: box labels come from the PNG grid
        box = {
            "x": [self.xmin, self.xmax],
            "y": [self.ymin, self.ymax],
            "label": [self.left_label, self.right_label, self.bottom_label, self.top_label],
        }

        scheme = {
            "velocities": list(range(9)),
            "conserved_moments": [self.rho, self.qx, self.qy],
            "polynomials": [
                1, self.LA * self.X, self.LA * self.Y,
                3 * (self.X**2 + self.Y**2) - 4,
                (9 * (self.X**2 + self.Y**2)**2 - 21 * (self.X**2 + self.Y**2) + 8) / 2,
                3 * self.X * (self.X**2 + self.Y**2) - 5 * self.X,
                3 * self.Y * (self.X**2 + self.Y**2) - 5 * self.Y,
                self.X**2 - self.Y**2,
                self.X * self.Y,
            ],
            "relaxation_parameters": self.s,
            "equilibrium": [
                self.rho, self.qx, self.qy,
                -2 * self.rho + 3 * self.q2,
                self.rho - 3 * self.q2,
                -self.qx / self.LA, -self.qy / self.LA,
                self.qx2 - self.qy2, self.qxy,
            ],
        }

        init = {self.rho: self.rho0, self.qx: 0.0, self.qy: 0.0}

        # Map grid labels to pylbm BCs
        bcs = {
            self.L_INLET: {
                "method": {0: pylbm.bc.BouzidiBounceBack},
                "value": inlet_momentum_bc,
            },
            self.L_WALL: {
                "method": {0: pylbm.bc.BouzidiBounceBack},
            },
            self.L_OUTLET: {
                "method": {0: pylbm.bc.NeumannX},
            },
        }

        return {
            "box": box,
            "elements": self.elements,
            "space_step": self.dx,
            "scheme_velocity": self.la,
            "parameters": {self.LA: self.la},
            "schemes": [scheme],
            "init": init,
            "boundary_conditions": bcs,
            "generator": "cython",
        }

    # --------------------------------------------------
    # Run + Plot
    # --------------------------------------------------
    def run(self):
        self.sol = pylbm.Simulation(self.build_simulation_config())
        while self.sol.t < self.Tf:
            self.sol.one_time_step()

        return self.sol

    def plot(self):
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]

        p = self.pressure_field(self.sol)
        img = (p - p.mean()).T
        ax.image(img, cmap="viridis")

        ax.title = f"Pressure field at t = {self.sol.t:f}"
        fig.show()

    # Optional: visualize the parsed PNG grid itself
    def plot_grid(self):
        plot_grid(self.grid)

    def animate(self, nrep: int = 50, interval: int = 1):
        """
        Live matplotlib animation using the existing pylbm.Simulation.
        """
        import matplotlib.patches as patches

        if not hasattr(self, "sol"):
            raise RuntimeError("Simulation not built. Call build_solver() first.")
        
        sol = self.sol

        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]

        # obstacle_mask = np.zeros((self.W, self.H))
        # # for x0, x1, y0, y1 in self.rects:
        # #     obstacle_mask[x0:x1, y0:y1] = 1.0  # mark obstacle cells

        # # field_data = self.pressure_field(sol).T
        # # combined = field_data.copy()
        # # combined[obstacle_mask.T > 0] = field_data.max()  # or a fixed color
        # # ax.image(combined, cmap="viridis")

        # Initial field
        p = self.pressure_field(sol)
        image = ax.image((p - p.mean()).T, cmap="viridis")

        ax.title = f"Pressure field, t = {sol.t:.3f}"

        def update(frame):
            for _ in range(nrep):
                sol.one_time_step()

            p = self.pressure_field(sol)
            image.set_data((p - p.mean()).T)
            ax.title = f"Pressure field, t = {sol.t:.3f}"

        fig.animate(update, interval=1)

        plt.show()
    
    def draw_elements(self, ax, color="red", alpha=0.6):
        """
        Draw obstacles using pylbm matplotlib_viewer methods.
        """
        # draw rectangles
        for (x0, x1, y0, y1) in getattr(self, "rects", []):
            px = float(x0) * self.dx
            py = float(y0) * self.dx
            w = float(x1 - x0) * self.dx
            h = float(y1 - y0) * self.dx
            ax.rectangle([px, py], [w, h], color=color, alpha=alpha)

        # draw circles
        for elem in self.elements:
            if isinstance(elem, pylbm.Circle):
                cx, cy = elem.center
                r = elem.radius
                ax.ellipse([cx, cy], [r, r], color=color, alpha=alpha)



if __name__ == "__main__":
    sim = Simulation(dt = 0.1, png_path="./data/test.png")
    # sim.plot_grid()  # uncomment to debug your PNG -> grid parsing
    sol = sim.run()
    print("Flow resistance R =", sim.flow_resistance())
    sim.animate(nrep=1)
