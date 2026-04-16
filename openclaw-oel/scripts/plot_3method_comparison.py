#!/usr/bin/env python3
"""Plot 4-method comparison: OPCD-Suc vs OPCD-Pre vs OPCD vs SDFT/SDPO-style.

Naming convention:
  - SDFT/SDPO-style: per-turn distillation without persistent experience (baseline)
  - OPCD:     On-Policy Context Distillation, cross-session experience accumulation
  - OPCD-Pre: OPCD + before-the-fact per-turn experience extraction
  - OPCD-Suc: OPCD + after-the-fact successive (post-hoc) extraction + replay

Data source: reproduction runs (2026-04-16).
Style: OPCD/OEL-style paper figure format.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# ── Load data from reproduction result JSONs ────────────────────────────
# Auto-detect the latest result file for each method, or use explicit paths.

import glob as _glob


def _find_latest(directory: str, pattern: str) -> str | None:
    """Find the most recently modified file matching pattern in directory."""
    matches = sorted(_glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
    return matches[-1] if matches else None


OEL_RESULTS_DIR = os.path.join(REPO_ROOT, "openclaw-oel", "eval", "results")
OPD_RESULTS_DIR = os.path.join(REPO_ROOT, "openclaw-opd", "eval", "results")

# Map method name → (results_dir, file_glob_pattern)
# reproduce_all.sh tags outputs as: personal_agent_oel_TIMESTAMP_oel_replay.json
_METHOD_SEARCH = {
    "OPCD-Suc (Ours)":    (OEL_RESULTS_DIR, "*_opcd_suc.json"),
    "OPCD-Pre":            (OEL_RESULTS_DIR, "*_opcd_pre.json"),
    "OPCD":                (OEL_RESULTS_DIR, "*_opcd.json"),
    "SDFT/SDPO-style":     (OPD_RESULTS_DIR, "*_sdpo.json"),
}

# Fallback: explicit paths from the original reproduction (2026-04-16)
_FALLBACK = {
    "OPCD-Suc (Ours)":    os.path.join(OEL_RESULTS_DIR, "personal_agent_oel_20260416_025307.json"),
    "OPCD-Pre":            os.path.join(OEL_RESULTS_DIR, "personal_agent_oel_20260416_042102.json"),
    "OPCD":                os.path.join(OEL_RESULTS_DIR, "personal_agent_oel_20260416_002055.json"),
    "SDFT/SDPO-style":     os.path.join(OPD_RESULTS_DIR, "personal_agent_opd_20260415_231252.json"),
}

RESULT_FILES = {}
for name, (rdir, pat) in _METHOD_SEARCH.items():
    found = _find_latest(rdir, pat)
    if found:
        RESULT_FILES[name] = found
    elif os.path.exists(_FALLBACK[name]):
        RESULT_FILES[name] = _FALLBACK[name]
    else:
        print(f"WARNING: no result file found for {name}")

if not RESULT_FILES:
    raise FileNotFoundError("No result files found. Run reproduce_all.sh first.")


def load_series(path):
    """Load evaluations from result JSON, return (rounds, student_scores, teacher_scores)."""
    with open(path) as f:
        d = json.load(f)
    evals = d["evaluations"]
    rounds, students, teachers = [], [], []
    for e in evals:
        rounds.append(e["round"])
        students.append(float(np.mean(e["student"]["scores"])))
        teachers.append(float(np.mean(e["teacher"]["scores"])))
    return rounds, students, teachers


data = {}
for name, path in RESULT_FILES.items():
    r, s, t = load_series(path)
    data[name] = {"rounds": r, "student": s, "teacher": t}

# ── Styles (OPCD/OEL paper style) ───────────────────────────────────────

styles = {
    "OPCD-Suc (Ours)":    {"color": "#E63946", "lw": 3.5, "zorder": 10},
    "OPCD-Pre":            {"color": "#E9C46A", "lw": 3.5, "zorder": 5},
    "OPCD":                {"color": "#457B9D", "lw": 3.5, "zorder": 5},
    "SDFT/SDPO-style":     {"color": "#2A9D8F", "lw": 3.5, "zorder": 5},
}

# ── Figure (two panels: Student + Teacher) ───────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

for name, d in data.items():
    s = styles[name]
    ax1.plot(d["rounds"], d["student"], label=name,
             color=s["color"], linewidth=s["lw"], zorder=s["zorder"], alpha=0.9)
    ax2.plot(d["rounds"], d["teacher"], label=name,
             color=s["color"], linewidth=s["lw"], zorder=s["zorder"], alpha=0.9)

for ax, ylabel, title, ylim in [
    (ax1, "Student Score", "Student Score (Learning)", (0.25, 0.48)),
    (ax2, "Teacher Score", "Teacher Score (Anti-Forgetting)", (0.45, 0.78)),
]:
    ax.set_xlabel("Training Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(ylim)
    ax.set_xlim(-1, 42)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.axhline(y=0.3 if "Student" in ylabel else 0.55, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    leg = ax.legend(loc="best", framealpha=0.85)
    for text in leg.get_texts():
        if "(Ours)" in text.get_text():
            text.set_fontweight("bold")

fig.suptitle("Qwen3-1.7B, 40 Rounds — 4-Method Comparison",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()

out_path = os.path.join(SCRIPT_DIR, "3method_comparison.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out_path}")
plt.close(fig)
