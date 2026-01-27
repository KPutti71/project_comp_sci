import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_3d_and_heatmap_full_sweep(
    csv_path,
    output_base="results/diodicity"
):
    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(csv_path)

    # Column names (edit if needed)
    x_col, y_col, z_col = df.columns[:3]

    # -----------------------
    # Pivot full sweep to grid
    # -----------------------
    Z_df = df.pivot(index=y_col, columns=x_col, values=z_col)

    X_vals = Z_df.columns.values
    Y_vals = Z_df.index.values
    Z = Z_df.values

    X, Y = np.meshgrid(X_vals, Y_vals)

    # -----------------------
    # Exact max ridge (per radius)
    # -----------------------
    ridge_x = []
    ridge_y = []
    ridge_z = []

    for y in Y_vals:
        x_max = Z_df.loc[y].idxmax()
        z_max = Z_df.loc[y, x_max]
        ridge_x.append(x_max)
        ridge_y.append(y)
        ridge_z.append(z_max)

    ridge_x = np.array(ridge_x)
    ridge_y = np.array(ridge_y)
    ridge_z = np.array(ridge_z)

    # =======================
    # 3D SURFACE PLOT
    # =======================
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title("3D plot of diodicity vs radius and dy")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_base}_surface_3d.png", dpi=300)
    plt.close(fig)

    # =======================
    # TOP-DOWN HEATMAP
    # =======================
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[X_vals.min(), X_vals.max(), Y_vals.min(), Y_vals.max()],
        cmap="viridis"
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Heatmap of diodicity vs radius and dy")

    plt.colorbar(im, ax=ax, label=z_col)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_base}_heatmap_topdown.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_3d_and_heatmap_full_sweep("results/diodicity_surface.csv")
