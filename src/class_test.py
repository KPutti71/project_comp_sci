import numpy as np
import sympy as sp
import pylbm
import matplotlib.pyplot as plt

from .png_to_grid import png_to_grid, plot_grid
from .valve_generator import generate_valve


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
        Tf: float = 1000.0,
        rho0: float = 1,
        mu_bulk: float = 1e-3,
        dt: int = 1,
        input_vel: int = 0.1,
        visualization_by: str = 'pressure_field',
        image_resolution: tuple = (200, 100),
        flip_y: bool = False,
        flip_x: bool = False
    ):
        # save simulation on instance
        self.sol = None

        self.dt = dt
        self.vis_method_str = visualization_by if hasattr(self, visualization_by) else "pressure_field"
        self.vis_method = getattr(self, self.vis_method_str)

        # --------------------------------------------------
        # 1) Load grid (store everything you’ll need on self)
        # --------------------------------------------------
        self.png_path = png_path
        self.grid = png_to_grid(self.png_path, resolution=image_resolution, flip_x=flip_x, flip_y=flip_y)

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

        # Inlet velocity
        self.u_in = input_vel

        # --------------------------------------------------
        # 3) Build pylbm obstacles from grid rectangles
        # --------------------------------------------------
        self.elements = self._build_elements_from_rects(self.rects)


        # ----
        # MEASUREMENTS BABY
        # ----

        # flow resistance measurement coords
        self.measure_in_y, self.measure_out_y = self.set_measure_y()
        self.measure_in_x, self.measure_out_x = self.set_measure_x()


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

    # functions to set measurement points for flow resistance measures

    def set_measure_y(self):
        """
        Function to find the coordinates of the measurement points.
        coordinates will be 
        """

        y_in = 0
        # find y_in top boundary
        while self.mask[:, 0][y_in] == 1 and y_in < self.ymax:
            y_in += 1
        
        # find diameter
        valve_diameter =0
        while self.mask[:, 0][y_in+valve_diameter] == 0 and y_in < self.ymax:
            valve_diameter += 1
        
        # find y-out top boundary
        y_out = 0
        while self.mask[:, -1][y_out] and y_out < self.ymax:
            y_out += 1

        y_in = y_in+(valve_diameter/2)
        y_out = y_out+(valve_diameter/2)
        
        return (int(y_in), int(y_out))

    def set_measure_x(self):
        # find y_in top boundary
        y_in = 0
        while self.mask[:, 0][y_in] == 1 and y_in < self.ymax:
            y_in += 1
        
        # find diameter
        valve_diameter =0
        while self.mask[:, 0][y_in+valve_diameter] == 0 and y_in < self.ymax:
            valve_diameter += 1

        # find first loop start x coord
        first_loop_x = 0
        while (self.mask[y_in - 1, :][first_loop_x] == 1 
               and self.mask[y_in+valve_diameter, :][first_loop_x] == 1
               and first_loop_x < self.xmax):
            first_loop_x += 1
        
        # find y-out top boundary
        y_out = 0
        while self.mask[:, -1][y_out] and y_out < self.ymax:
            y_out += 1
        
        # find final loop end x coord
        final_loop_x = int(self.xmax) - 1
        while (self.mask[y_out - 1, :][final_loop_x] == 1
               and self.mask[y_out + valve_diameter, :][final_loop_x] == 1
               and final_loop_x >= 0):
            final_loop_x -= 1
        
        return (int(first_loop_x), int(final_loop_x))

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
    def pressure_field(self):
        cs2 = 1.0 / 3.0
        return cs2 * self.sol.m[self.rho]

    def velocity_magnitude(self):
        ux = sol.m[self.qx] / self.sol.m[self.rho]
        uy = sol.m[self.qy] / sol.m[self.rho]
        return np.sqrt(ux**2 + uy**2)

    def flow_resistance(self, x_offset_cells: int = 2):
        """
        Uses pressure at two x-locations (near inlet/outlet) at mid-height.
        NOTE: Assumes sol.m[rho] indexing is [ix, iy] in lattice-cell coordinates.
        """
        p = self.vis_method()

        p_in = p[self.measure_in_x, self.measure_in_y]
        p_out = p[self.measure_out_x, self.measure_out_y]

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

        p = self.vis_method
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

        # draw the walls
        self.draw_elements(ax)

        # draw measure points
        if hasattr(self, "measure_in_x"):
            ax.ax.scatter([self.measure_in_x, self.measure_out_x], 
                          [self.measure_in_y, self.measure_out_y], 
                          c = "red", marker="o", zorder = 20)

        # Initial field
        p = self.vis_method()
        image = ax.image((p - p.mean()).T, cmap="viridis")

        ax.title = f"{self.vis_method_str}, t = {sol.t:.3f}"

        def update(frame):
            for _ in range(nrep):
                sol.one_time_step()

            p = self.vis_method()
            image.set_data((p - p.mean()).T)
            ax.title = f"{self.vis_method_str}, t = {sol.t:.3f}"

        fig.animate(update, interval=1)

        plt.show()

    def draw_elements(self, ax):
        """
        Draw all obstacles (Parallelograms from self.elements) on the given matplotlib axis.
        """

        obstacle_mask = np.zeros((self.W, self.H))  # lattice size
        for x0, x1, y0, y1 in self.rects:
            obstacle_mask[x0:x1, y0:y1] = 1.0  # mark obstacles

        masked = np.ma.masked_where(obstacle_mask == 0, obstacle_mask)

        img = ax.image(masked.T, clim = [0,1], cmap="binary", alpha=1)
        img.set_zorder(10)


def diodicity(reverse, forward):
    return reverse / forward


if __name__ == "__main__":
    generate_valve(0.07, 2, 0.3, 0.1, 1)
    resolution = (400, 200)
    vis_method = "velocity_magnitude"

    # Run simulations
    sim = Simulation(la = 1.03, input_vel=0.020, png_path="data/valve.png", flip_x=False,
                      image_resolution=resolution, visualization_by=vis_method)
    sim_rev = Simulation(la = 1.03, input_vel=0.020, png_path="data/valve.png", flip_x=True,
                         image_resolution=resolution, visualization_by=vis_method)
    # sim.plot_grid()  # uncomment to debug your PNG -> grid parsing
    sol = sim.run()
    sol_rec = sim_rev.run()

    # Calculate flow resistance for diodicity
    reverse_res = sim_rev.flow_resistance()
    forward_res = sim.flow_resistance()
    print(f"Reverse Flow resistance at time t = {sim_rev.Tf} is R = {reverse_res}")
    print(f"Forward Flow resistance at time t = {sim.Tf} is R = {forward_res}")
    Diodicity = diodicity(reverse_res, forward_res)
    print(f"The diodicity of this pipe is Di = {Diodicity}")

    # Run animation of reverse flow
    sim.animate(nrep=10, interval = 1000)


"""
animation parameters
nrep = stepsize of t per frame. nrep = 60 -> everyframe t += 60
interval = the amount of ms between frames

simulation parameters
la = 1.02, input_vel=0.015 works pretty well
la = is scheme velocity. Increasing this will make the simulation more stable, but simulation time longer.
input_vel: input velocity of the fluid. Decrease for more stable simulation.

"""
