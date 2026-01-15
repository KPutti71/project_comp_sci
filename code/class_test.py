import sympy as sp
import pylbm


class Simulation:
    def __init__(self, obj=None):
        self.obj = obj

        self.xmin, self.xmax = 0.0, 3.0
        self.ymin, self.ymax = 0.0, 1.0
        self.radius = 0.2

        self.Re = 20
        self.dx = 1.0 / 128
        self.la = 1.0
        self.Tf = 75
        self.rho0 = 1.0
        self.mu_bulk = 1e-3

        # Symbolic variables
        self.X, self.Y, self.LA = sp.symbols("X Y LA")
        self.rho, self.qx, self.qy = sp.symbols("rho qx qy")

        # Derived parameters
        self.cylinder_center = [0.3, 0.5 * (self.ymin + self.ymax) + self.dx]
        self.u_in = self.la / 20.0
        self.eta_shear = self.rho0 * self.u_in * (2.0 * self.radius) / self.Re

        dummy = 3.0 / (self.la * self.rho0 * self.dx)
        s_mu = 1.0 / (0.5 + self.mu_bulk * dummy)
        s_eta = 1.0 / (0.5 + self.eta_shear * dummy)
        self.s = [0.0, 0.0, 0.0, s_mu, s_mu, s_eta, s_eta, s_eta, s_eta]

        inv = 1.0 / (self.LA**2 * self.rho0)
        self.qx2 = inv * self.qx**2
        self.qy2 = inv * self.qy**2
        self.q2 = self.qx2 + self.qy2
        self.qxy = inv * self.qx * self.qy

    def pressure_field(self, sol):
        cs2 = 1.0 / 3.0
        return cs2 * sol.m[self.rho]

    def flow_resistance(self, sol):
        p = self.pressure_field(sol)
        mid_y = int((self.ymin + self.ymax) / 2.0)
        p_in = p[int(self.xmin + 2), mid_y]
        p_out = p[int(self.xmax - 2), mid_y]
        Q = self.u_in * (self.ymax - self.ymin)
        return abs(p_in - p_out) / Q

    def build_simulation_config(self):
        # --- IMPORTANT FIX: use a plain function (closure), not a bound method ---
        qx_sym = self.qx
        rho0 = self.rho0
        u_in = self.u_in

        def inlet_momentum_bc(f, m, x, y):
            m[qx_sym] = rho0 * u_in

        box = {"x": [self.xmin, self.xmax], "y": [self.ymin, self.ymax], "label": [0, 2, 0, 0]}
        elements = [pylbm.Circle(self.cylinder_center, self.radius, label=1)]

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

        bcs = {
            0: {"method": {0: pylbm.bc.BouzidiBounceBack}, "value": inlet_momentum_bc},
            1: {"method": {0: pylbm.bc.BouzidiBounceBack}},
            2: {"method": {0: pylbm.bc.NeumannX}},
        }

        return {
            "box": box,
            "elements": elements,
            "space_step": self.dx,
            "scheme_velocity": self.la,
            "parameters": {self.LA: self.la},
            "schemes": [scheme],
            "init": init,
            "boundary_conditions": bcs,
            "generator": "cython",
        }

    def run(self):
        sol = pylbm.Simulation(self.build_simulation_config())
        while sol.t < self.Tf:
            sol.one_time_step()
        return sol

    def plot(self, sol):
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]

        p = self.pressure_field(sol)
        img = (p - p.mean()).T
        ax.image(img, cmap="viridis")

        ax.ellipse(
            [self.cylinder_center[0] / self.dx, (0.5 * (self.ymin + self.ymax)) / self.dx],
            [self.radius / self.dx, self.radius / self.dx],
            "r",
        )

        ax.title = f"Pressure field at t = {sol.t:f}"
        fig.show()


if __name__ == "__main__":
    sim = Simulation()
    sol = sim.run()
    print("Flow resistance R =", sim.flow_resistance(sol))
    sim.plot(sol)
