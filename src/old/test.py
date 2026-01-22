import numpy as np
import matplotlib.pyplot as plt

nx, ny = (10, 10)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)

density = np.random.rand(nx, ny)

def diffuse(M, k=0.3):
    N = np.zeros_like(M)
    nx, ny = M.shape

    for (i, j), d in np.ndenumerate(M):
        s = (
            M[(i - 1) % nx, j] +
            M[i, (j - 1) % ny] +
            M[(i + 1) % nx, j] +
            M[i, (j + 1) % ny]
        ) / 4.0
        N[i, j] = d + k * (s - d)

    return N

# plt.plot(xv, yv, marker='o', color='k', linestyle='none')

plt.imshow(density)
plt.show()
for _ in range(10):
    density = diffuse(density)
    plt.imshow(density)
    plt.show()

