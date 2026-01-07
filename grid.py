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
            gradient_hori[i] = -self.hori[i] + self.hori[i+1]
        return gradient_hori


def main():
    grid = StaggeredGrid(10, 10)
    grid.random_values()

    xv, yv = grid.cell_centers()
    hori_xv, hori_yv, vert_xv, vert_yv = grid.edge_points()

    # fig, ax = plt.subplots()
    # ax.plot(xv, yv, "ko")
    # ax.plot(hori_xv, hori_yv, "ro")
    # ax.plot(vert_xv, vert_yv, "go")
    # ax.set_aspect("equal")
    # plt.show()

    cells = grid.gradient_cells()
    print(cells)
    plt.imshow(cells)
    plt.show()


if __name__ == "__main__":
    main()
