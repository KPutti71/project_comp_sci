import numpy as np
import matplotlib.pyplot as plt


class StaggeredGrid:
    def __init__(self, nx=10, ny=10):
        self.nx = int(nx)
        self.ny = int(ny)

        self.cell = np.zeros((self.ny, self.nx))
        self.hori = np.zeros((self.ny + 1, self.nx))
        self.vert = np.zeros((self.ny, self.nx + 1))
    
    def random_values(self):
        self.hori = (np.random.rand(self.ny + 1, self.nx) - 0.5) * 2
        self.vert = np.random.rand(self.ny, self.nx + 1)

    def cell_centers(self):
        cx = np.arange(self.nx) + 0.5
        cy = np.arange(self.ny) + 0.5
        return np.meshgrid(cx, cy)

    def edge_points(self):
        cx = np.arange(self.nx) + 0.5
        cy = np.arange(self.ny) + 0.5

        hori_xv, hori_yv = np.meshgrid(cx, np.arange(self.ny + 1))
        vert_xv, vert_yv = np.meshgrid(np.arange(self.nx + 1), cy)

        return hori_xv, hori_yv, vert_xv, vert_yv
    
    def gradient_cells(self):
        gradient_hori = np.zeros((self.ny, self.nx))
        for i in range(self.ny):
            gradient_hori[i, :] = self.hori[i + 1, :] - self.hori[i, :]

        gradient_vert = np.zeros((self.ny, self.nx))
        for j in range(self.nx):
            gradient_vert[:, j] = self.vert[:, j + 1] - self.vert[:, j]

        return -(gradient_hori + gradient_vert)
    
    def plot(self):
        hori_xv, hori_yv, vert_xv, vert_yv = self.edge_points()
        gradient = self.gradient_cells()

        v_h = self.hori
        u_v = self.vert

        max_mag = max(np.max(np.abs(v_h)), np.max(np.abs(u_v)))
        v_h_scaled = 0.5 * v_h / max_mag
        u_v_scaled = 0.5 * u_v / max_mag

        fig, ax = plt.subplots()

        # heatmap of the flow in/out of a cell
        im = ax.imshow(
            gradient,
            origin="lower",
            extent=(0, self.nx, 0, self.ny),
            cmap="viridis",
            alpha=0.8
        )

        # vertical quivers for the horizontal edges
        ax.quiver(
            hori_xv,
            hori_yv,
            np.zeros_like(v_h_scaled),
            v_h_scaled,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red"
        )

        # horizontal quivers for the vertical edges
        ax.quiver(
            vert_xv,
            vert_yv,
            u_v_scaled,
            np.zeros_like(u_v_scaled),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red"
        )

        # Makes the grid equally spaced
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0, self.nx + 1, 1))
        ax.set_yticks(np.arange(0, self.ny + 1, 1))
        ax.grid(True)

        plt.colorbar(im, ax=ax)
        plt.show()


def main():
    grid = StaggeredGrid(10, 10)
    grid.random_values()
    grid.plot()


if __name__ == "__main__":
    main()
