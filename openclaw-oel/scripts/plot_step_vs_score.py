#!/usr/bin/env python3
"""Plot round vs score for all 5 methods (Student & Teacher).

X-axis: Training Round (0–40), the fairest comparison since all methods
ran exactly 40 rounds. Different methods produce different numbers of
weight-update steps per round, so using round number normalises that.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
})

# ── Data (keyed by round number) ─────────────────────────────────────────

data = {
    "OEL v2": {
        "rounds":  [0,  6, 10, 15, 20, 23, 26, 28, 31, 34, 37, 39, 40],
        "student": [0.315, 0.300, 0.345, 0.329, 0.435, 0.347, 0.334, 0.372, 0.347, 0.370, 0.393, 0.441, 0.460],
        "teacher": [0.577, 0.610, 0.761, 0.692, 0.535, 0.454, 0.492, 0.487, 0.413, 0.423, 0.467, 0.448, 0.279],
    },
    "Combined": {
        "rounds":  [0,  7, 12, 16, 20, 24, 29, 33, 36, 40],
        "student": [0.283, 0.342, 0.331, 0.299, 0.315, 0.319, 0.353, 0.353, 0.357, 0.407],
        "teacher": [0.555, 0.667, 0.622, 0.635, 0.606, 0.686, 0.627, 0.617, 0.670, 0.613],
    },
    "OPD": {
        "rounds":  [0, 10, 16, 25, 33, 37, 40],
        "student": [0.304, 0.294, 0.351, 0.341, 0.341, 0.390, 0.353],
        "teacher": [0.534, 0.586, 0.684, 0.728, 0.654, 0.579, 0.648],
    },
    "OEL v1": {
        "rounds":  [0,  6, 11, 15, 19, 21, 24, 27, 30, 34, 37, 40],
        "student": [0.296, 0.291, 0.341, 0.357, 0.363, 0.369, 0.379, 0.307, 0.330, 0.322, 0.319, 0.325],
        "teacher": [0.590, 0.598, 0.752, 0.639, 0.596, 0.583, 0.591, 0.577, 0.594, 0.531, 0.535, 0.569],
    },
    "RL (GRPO)": {
        "rounds":  [0,  5,  8, 11, 14, 17, 19, 22, 25, 28, 31, 34, 36, 40],
        "student": [0.278, 0.331, 0.311, 0.303, 0.289, 0.311, 0.274, 0.283, 0.278, 0.269, 0.165, 0.157, 0.100, 0.059],
        "teacher": [0.523, 0.579, 0.560, 0.467, 0.492, 0.485, 0.423, 0.442, 0.398, 0.311, 0.286, 0.215, 0.194, 0.160],
    },
}

# Colors & markers
styles = {
    "OEL v2":     {"color": "#E63946", "marker": "o",  "lw": 2.5, "zorder": 10},
    "Combined":   {"color": "#457B9D", "marker": "s",  "lw": 2.0, "zorder": 5},
    "OPD":        {"color": "#2A9D8F", "marker": "^",  "lw": 2.0, "zorder": 5},
    "OEL v1":     {"color": "#E9C46A", "marker": "D",  "lw": 2.0, "zorder": 5},
    "RL (GRPO)":  {"color": "#999999", "marker": "v",  "lw": 1.8, "zorder": 3},
}


def plot_panel(ax, score_key, title, ylabel, ylim):
    for name, d in data.items():
        s = styles[name]
        ax.plot(
            d["rounds"], d[score_key],
            label=name,
            color=s["color"], marker=s["marker"], markersize=6,
            linewidth=s["lw"], zorder=s["zorder"], alpha=0.9,
        )
    ax.set_xlabel("Training Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(ylim)
    ax.set_xlim(-1, 42)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.axhline(y=0.3, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.85)


# ── Figure ────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

plot_panel(ax1, "student", "Student Score vs. Round", "Student Score", (0.0, 0.52))
plot_panel(ax2, "teacher", "Teacher Score vs. Round", "Teacher Score", (0.05, 0.82))

fig.suptitle(
    "OpenClaw-RL: 5-Method Comparison on Hard GSM8K (Qwen3-1.7B, 40 Rounds)",
    fontsize=15, fontweight="bold", y=1.01,
)
fig.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "step_vs_score.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out_path}")
plt.close(fig)
