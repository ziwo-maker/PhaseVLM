import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# Data
# =========================
rows = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
cols = [r"$t-T+1$", "", r"$\cdots$", r"$t-1$", r"$t$"]

data = [
    ["3", "5", r"$\cdots$", "2", "4"],
    ["1", "2", r"$\cdots$", "0", "1"],
    ["0", "3", r"$\cdots$", "3", "2"],
    ["2", "4", r"$\cdots$", "8", "3"],
    ["3", "2", r"$\cdots$", "3", "2"],
    ["2", "3", r"$\cdots$", "2", "3"],
    ["1", "2", r"$\cdots$", "0", "1"],
    ["3", "3", r"$\cdots$", "2", "3"],
]

# =========================
# Figure settings
# =========================
fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
ax.set_xlim(-0.8, 5.1)
ax.set_ylim(-0.2, 10.1)
ax.axis("off")

# Colors
cell_bg = "#eaf4ff"
cell_bg_alt = "#dcecff"
grid_color = "#5b8fc1"
text_color = "#111111"

# Table geometry
n_rows = len(rows)
n_cols = len(cols)

cell_w = 1.0
cell_h = 0.9
x0 = 0.0
y0 = 0.8

# =========================
# Title and formula
# =========================
ax.text(
    2.5, 9.55,
    r"Temporal Window (length $T$)",
    ha="center", va="center",
    fontsize=17, fontweight="bold"
)

ax.text(
    2.5, 8.75,
    r"$X_t \in \mathbb{R}^{T \times D}$",
    ha="center", va="center",
    fontsize=18, fontweight="bold"
)

# =========================
# Column labels
# =========================
for j, col in enumerate(cols):
    ax.text(
        x0 + j * cell_w + cell_w / 2,
        y0 + n_rows * cell_h + 0.42,
        col,
        ha="center", va="center",
        fontsize=14, fontweight="bold"
    )

# =========================
# Draw cells
# =========================
for i in range(n_rows):
    for j in range(n_cols):
        x = x0 + j * cell_w
        y = y0 + (n_rows - 1 - i) * cell_h

        bg = cell_bg if j != 3 else cell_bg_alt

        ax.add_patch(
            Rectangle(
                (x, y),
                cell_w,
                cell_h,
                facecolor=bg,
                edgecolor=grid_color,
                linewidth=1.2
            )
        )

        ax.text(
            x + cell_w / 2,
            y + cell_h / 2,
            data[i][j],
            ha="center", va="center",
            fontsize=14,
            color=text_color,
            fontweight="bold" if data[i][j] != r"$\cdots$" else "normal"
        )

# =========================
# Row labels
# =========================
for i, row in enumerate(rows):
    y = y0 + (n_rows - 1 - i) * cell_h + cell_h / 2
    ax.text(
        x0 - 0.35,
        y,
        row,
        ha="right", va="center",
        fontsize=14,
        fontweight="bold"
    )

# Outer border
ax.add_patch(
    Rectangle(
        (x0, y0),
        n_cols * cell_w,
        n_rows * cell_h,
        fill=False,
        edgecolor=grid_color,
        linewidth=1.8
    )
)

# =========================
# Save
# =========================
plt.tight_layout()
plt.savefig("temporal_window.png", dpi=600, bbox_inches="tight")
plt.savefig("temporal_window.pdf", bbox_inches="tight")
plt.savefig("temporal_window.svg", bbox_inches="tight")
plt.show()
