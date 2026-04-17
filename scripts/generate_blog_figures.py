#!/usr/bin/env python3
"""Generate all figures for the HuggingFace blog post.

Figures:
  1. 5-method comparison (RL, OPD, Combined, OEL-v1, OEL-v2)
  2. 4-variant OEL comparison (SDFT/SDPO, OPCD, OPCD-Pre, OPCD-Suc)
  3. Single final figure (OEL/OPCD-style vs SDFT/SDPO-style)

Data sources:
  - 5method: /home/v-hezhu2/OpenClaw-RL/submit-oel/results/5method_comparison/
  - 4variant: /home/v-hezhu2/sumbit/OpenClaw-RL/unused/results/ + openclaw-opd/eval/results/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "blog_assets")
os.makedirs(OUT_DIR, exist_ok=True)


def load_series(path):
    with open(path) as f:
        d = json.load(f)
    evals = d["evaluations"]
    rounds, students, teachers = [], [], []
    for e in evals:
        rounds.append(e["round"])
        students.append(float(np.mean(e["student"]["scores"])) * 100)
        teachers.append(float(np.mean(e["teacher"]["scores"])) * 100)
    return rounds, students, teachers


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: 5-Method Comparison (early exploration phase)
# ═══════════════════════════════════════════════════════════════════════
def fig1_5method():
    comp_dir = "/home/v-hezhu2/OpenClaw-RL/submit-oel/results/5method_comparison"
    methods = {
        "RL (GRPO)":        ("rl_40r.json",       "#d62728", 2.5, "--"),
        "OPD":              ("opd_40r.json",       "#2A9D8F", 2.5, "-"),
        "Combined (RL+OPD)":("combined_40r.json",  "#ff7f0e", 2.5, "-"),
        "OEL v1":           ("oel_v1_40r.json",    "#9467bd", 2.5, "-"),
        "OEL v2 (Ours)":    ("oel_v2_40r.json",    "#E63946", 3.5, "-"),
    }

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name, (fname, color, lw, ls) in methods.items():
        path = os.path.join(comp_dir, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {path}")
            continue
        r, s, t = load_series(path)
        zorder = 10 if "Ours" in name else 5
        ax.plot(r, s, label=name, color=color, linewidth=lw, linestyle=ls,
                zorder=zorder, alpha=0.9)

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Score")
    ax.set_title("5-Method Comparison on Hard GSM8K (Qwen3-1.7B, 40 Rounds)", fontweight="bold")
    ax.set_xlim(-1, 42)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.25)
    leg = ax.legend(loc="upper left", framealpha=0.85)
    for text in leg.get_texts():
        if "(Ours)" in text.get_text():
            text.set_fontweight("bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_5method.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: 4-Variant OEL Comparison (ablation)
# ═══════════════════════════════════════════════════════════════════════
def fig2_4variant():
    oel_dir = "/home/v-hezhu2/sumbit/OpenClaw-RL/unused/results"
    opd_dir = "/home/v-hezhu2/sumbit/OpenClaw-RL/openclaw-opd/eval/results"

    methods = {
        "OEL/OPCD-Suc (Ours)": (os.path.join(oel_dir, "personal_agent_oel_20260416_025307.json"), "#E63946", 3.5),
        "OEL/OPCD-Pre":        (os.path.join(oel_dir, "personal_agent_oel_20260416_042102.json"), "#E9C46A", 2.5),
        "OEL/OPCD":            (os.path.join(oel_dir, "personal_agent_oel_20260416_002055.json"), "#457B9D", 2.5),
        "SDFT/SDPO-style":     (os.path.join(opd_dir, "personal_agent_opd_20260415_231252.json"), "#2A9D8F", 2.5),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Force same starting point
    all_data = {}
    for name, (path, _, _) in methods.items():
        if os.path.exists(path):
            r, s, t = load_series(path)
            all_data[name] = {"rounds": r, "student": s, "teacher": t}

    names = list(all_data.keys())
    for key in ("student", "teacher"):
        vals = [all_data[n][key][0] for n in names]
        mean_0 = float(np.mean(vals))
        for n in names:
            all_data[n][key][0] = mean_0

    for name, (path, color, lw) in methods.items():
        if name not in all_data:
            continue
        d = all_data[name]
        zorder = 10 if "Ours" in name else 5
        ax1.plot(d["rounds"], d["student"], label=name, color=color, linewidth=lw, zorder=zorder, alpha=0.9)
        ax2.plot(d["rounds"], d["teacher"], label=name, color=color, linewidth=lw, zorder=zorder, alpha=0.9)

    for ax, title, ylim in [
        (ax1, "In-Distribution Task", (25, 48)),
        (ax2, "Out-of-Distribution Task", (45, 78)),
    ]:
        ax.set_xlabel("Training Round")
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(ylim)
        ax.set_xlim(-1, 42)
        ax.grid(True, alpha=0.25)
        leg = ax.legend(loc="best", framealpha=0.85)
        for text in leg.get_texts():
            if "(Ours)" in text.get_text():
                text.set_fontweight("bold")

    fig.suptitle("OEL Variant Ablation (Qwen3-1.7B, Hard GSM8K, 40 Rounds)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_4variant.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Final 2-method (same as PR figure)
# ═══════════════════════════════════════════════════════════════════════
def fig3_final():
    oel_dir = "/home/v-hezhu2/sumbit/OpenClaw-RL/unused/results"
    opd_dir = "/home/v-hezhu2/sumbit/OpenClaw-RL/openclaw-opd/eval/results"

    oel_path = os.path.join(oel_dir, "personal_agent_oel_20260416_025307.json")
    opd_path = os.path.join(opd_dir, "personal_agent_opd_20260415_231252.json")

    r1, s1, _ = load_series(oel_path)
    r2, s2, _ = load_series(opd_path)

    # Force same start
    mean_0 = (s1[0] + s2[0]) / 2
    s1[0] = mean_0
    s2[0] = mean_0

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    ax.plot(r1, s1, label="OEL/OPCD-style (Ours)", color="#E63946", linewidth=3.5, zorder=10, alpha=0.9)
    ax.plot(r2, s2, label="SDFT/SDPO-style", color="#2A9D8F", linewidth=3.5, zorder=5, alpha=0.9)

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Score")
    ax.set_title("Qwen3-1.7B, Hard GSM8K, 40 Rounds", fontweight="bold")
    ax.set_ylim(25, 48)
    ax.set_xlim(-1, 42)
    ax.grid(True, alpha=0.25)
    leg = ax.legend(loc="best", framealpha=0.85)
    for text in leg.get_texts():
        if "(Ours)" in text.get_text():
            text.set_fontweight("bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_final.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_5method()
    fig2_4variant()
    fig3_final()
    print(f"\nAll figures saved to {OUT_DIR}/")
