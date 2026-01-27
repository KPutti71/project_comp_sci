import pandas as pd
import matplotlib.pyplot as plt

plt.figure()

cmap = plt.get_cmap("viridis")
n = len(diodicities)

for i, (t, d) in enumerate(zip(timepoints, diodicities)):
    if len(t) == 0:
        continue
    plt.plot(
        t,
        d,
        color=cmap(i / n),
        label=f"w={widths[i//4]}, r={radii[(i//2)%2]}, dy={dys[i%2]}",
    )

plt.xlabel("time")
plt.ylabel("diodicity")
plt.title("Diodicity over time for 8 valve configurations")
plt.ylim((0, 5))
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()