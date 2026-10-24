import numpy as np
import matplotlib.pyplot as plt

# Keep text editable in SVG
plt.rcParams["svg.fonttype"] = "none"

# -----------------------------
# Channel 1: Magnitude
# -----------------------------
movements = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

data = np.array([
    [0.25, 0.30, 0.55, 0.75, 0.80, 0.78, 0.72, 0.45],
    [0.18, 0.28, 0.50, 0.52, 0.50, 0.48, 0.46, 0.35],
    [0.32, 0.45, 0.70, 0.58, 0.62, 0.78, 0.50, 0.38],
    [0.20, 0.34, 0.48, 0.60, 0.64, 0.58, 0.42, 0.30],
    [0.26, 0.46, 0.62, 0.78, 0.66, 0.62, 0.45, 0.36],
    [0.22, 0.38, 0.58, 0.82, 0.70, 0.75, 0.72, 0.48],
    [0.28, 0.36, 0.60, 0.76, 0.64, 0.58, 0.56, 0.44],
    [0.24, 0.34, 0.52, 0.58, 0.55, 0.42, 0.40, 0.42],
])

fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)

im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

ax.set_yticks(np.arange(len(movements)))
ax.set_yticklabels(movements, fontsize=11, fontweight="bold")
ax.set_xticks([])

ax.set_title("Channel 1: Magnitude (Traffic Count)",
             fontsize=14, fontweight="bold", pad=12)

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
plt.savefig("channel_1_magnitude.svg", format="svg", bbox_inches="tight")
plt.show()
