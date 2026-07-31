import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Keep text editable in SVG
plt.rcParams["svg.fonttype"] = "none"

# -----------------------------
# Channel 3: Slow-varying Trend
# -----------------------------
movements = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

data = np.array([
    [0.18, 0.22, 0.42, 0.46, 0.68, 0.70, 0.76, 0.74],
    [0.16, 0.20, 0.34, 0.36, 0.48, 0.62, 0.72, 0.70],
    [0.14, 0.18, 0.28, 0.30, 0.36, 0.44, 0.52, 0.76],
    [0.12, 0.15, 0.24, 0.26, 0.72, 0.50, 0.58, 0.78],
    [0.10, 0.13, 0.22, 0.24, 0.38, 0.52, 0.66, 0.74],
    [0.08, 0.12, 0.20, 0.23, 0.32, 0.46, 0.55, 0.76],
    [0.06, 0.10, 0.18, 0.22, 0.30, 0.44, 0.68, 0.80],
    [0.05, 0.09, 0.16, 0.20, 0.28, 0.40, 0.70, 0.78],
])

cmap = LinearSegmentedColormap.from_list(
    "blue_white_yellow",
    ["#0B4DFF", "#4F83FF", "#EAEFFB", "#FFF16A", "#FFD400"]
)

fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)

im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

ax.set_yticks(np.arange(len(movements)))
ax.set_yticklabels(movements, fontsize=11, fontweight="bold")
ax.set_xticks([])

ax.set_title("Channel 3: Slow-varying Trend\n(Smoothed Trend)",
             fontsize=14, fontweight="bold", pad=10)

ax.set_ylabel("Movements", fontsize=12, fontweight="bold", labelpad=18)

ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.45)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(axis="y", length=0)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color("0.55")

ax.text(0.00, -0.18, "Past", transform=ax.transAxes,
        ha="left", va="center", fontsize=12, fontweight="bold")
ax.text(0.50, -0.18, "Time", transform=ax.transAxes,
        ha="center", va="center", fontsize=12, fontweight="bold")
ax.annotate(
    "", xy=(0.80, -0.18), xytext=(0.58, -0.18),
    xycoords=ax.transAxes,
    arrowprops=dict(arrowstyle="->", lw=1.6, color="black")
)
ax.text(1.00, -0.18, "Now", transform=ax.transAxes,
        ha="right", va="center", fontsize=12, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.08)
cbar.set_ticks([])
cbar.outline.set_linewidth(0.8)
cbar.ax.text(1.8, 1.00, "High", transform=cbar.ax.transAxes,
             ha="left", va="center", fontsize=11, fontweight="bold")
cbar.ax.text(1.8, 0.00, "Low", transform=cbar.ax.transAxes,
             ha="left", va="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("channel_3_trend.svg", format="svg", bbox_inches="tight")
plt.show()
