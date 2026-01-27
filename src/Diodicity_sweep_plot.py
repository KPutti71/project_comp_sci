import pandas as pd
import matplotlib.pyplot as plt

def plot_8_parameter_sweeps(
    csv_path,
    output_path="results/diodicity_8panel.png",
):
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["diodicity"])

    # -----------------------------
    # Identify unique parameter sets
    # -----------------------------
    param_cols = ["width", "radius", "dy"]
    param_sets = df[param_cols].drop_duplicates().iloc[:8]

    # -----------------------------
    # Create figure
    # -----------------------------
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    axes = axes.flatten()

    # -----------------------------
    # Plot each parameter set
    # -----------------------------
    for ax, (_, params) in zip(axes, param_sets.iterrows()):
        w, r, dy = params

        sub = df[
            (df["width"] == w) &
            (df["radius"] == r) &
            (df["dy"] == dy)
        ].sort_values("timepoint")

        ax.plot(sub["timepoint"], sub["diodicity"])
        ax.set_title(f"w={w}, r={r}, dy={dy}", fontsize=9)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Diodicity")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_8_parameter_sweeps("results/diodicity_sweep_all.csv")
