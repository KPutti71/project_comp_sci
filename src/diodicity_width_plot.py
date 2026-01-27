import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_diodicity_time(csv_path, output_base="results/diodicity"):
    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(csv_path)

    # Expect columns: width, radius, dy, timepoint, diodicity
    required_cols = {"width", "radius", "dy", "timepoint", "diodicity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Drop rows without data
    df = df.dropna(subset=["timepoint", "diodicity"])

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    groups = df.groupby(["width", "radius", "dy"])
    n = len(groups)
    cmap = plt.get_cmap("viridis")

    for i, ((width, radius, dy), g) in enumerate(groups):
        ax.plot(
            g["timepoint"].values,
            g["diodicity"].values,
            color=cmap(i / n),
            label=f"w={width}, r={radius}, dy={dy}",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("diodicity")
    ax.set_title("Diodicity over time for 8 valve configurations")
    ax.set_ylim((0, 5))
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_base}_time.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_diodicity_time("results/diodicity_sweep_all.csv")
