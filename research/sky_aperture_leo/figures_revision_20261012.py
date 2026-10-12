#!/usr/bin/env python3
"""Figures for the 2026-10-12 revision, drawn only from results/revision_20261012/*.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results/revision_20261012"
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, GRID = "#333333", "#dddddd"
C = ["#2a6fdb", "#e07b39", "#3a9e5f", "#a347b8"]


def fig_n3_wait():
    d = pd.read_csv(RES / "E0_baseline_all_waits.csv")
    d = d[(d.day == 1) & (d.config == "D3")].sort_values("wait_min")
    w = d.wait_min.to_numpy()
    show = d.valid_pct.to_numpy() >= 50          # median lies in the valid region
    show90 = d.valid_pct.to_numpy() >= 90
    fig, ax = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 1]))
    a = ax[0]
    a.plot(w[show], d.H_med[show], color=C[0], lw=1.8, label="水平CRLB 中央値")
    a.plot(w[show90], d.H_p90[show90], color=C[0], lw=1.2, ls="--", label="水平CRLB P90")
    a.plot(w[show], d.D_med[show], color=C[1], lw=1.8, label="3次元CRLB 中央値")
    a.axhline(100, color=INK, lw=0.8, ls=":")
    a.text(60, 100, "100 m", ha="right", va="bottom", fontsize=8, color=INK)
    a.set_yscale("log"); a.set_ylabel("CRLB [m]")
    a.grid(True, which="major", color=GRID, lw=0.6)
    a.legend(frameon=False, fontsize=8, loc="upper right")
    a.set_title("N=3 基準配置（1～60分、691開始時刻）", fontsize=9, loc="left")
    b = ax[1]
    b.plot(w, d.regular_pct, color=INK, lw=1.4, label="数値的正則率")
    b.plot(w, d.valid_pct, color=C[2], lw=1.8, label="有効率（3D CRLB≤1 km）")
    b.plot(w, d.A100, color=C[0], lw=1.8, label="水平CRLB≤100 m 達成率")
    b.set_ylim(0, 102); b.set_ylabel("割合 [%]"); b.set_xlabel("待ち時間 T [min]")
    b.grid(True, color=GRID, lw=0.6); b.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(RES / "fig_n3_wait.png", dpi=220); plt.close(fig)


def fig_azimuth():
    d = pd.read_csv(RES / "E2_azimuth_sweep.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for k, n in enumerate((3, 4, 5, 6)):
        s = d[d.N == n].sort_values("A0")
        ax.plot(s.A0, s.H_med, marker="o", ms=3, lw=1.5, color=C[k], label=f"N={n}")
        ax.plot([0], s.H_med.iloc[:1], marker="s", ms=7, mfc="none", color=C[k])
    ax.set_xlabel("開始方位 $A_0$ [deg]（各Nの周期 360°/N 内）")
    ax.set_ylabel("60分 水平CRLB 中央値 [m]")
    ax.set_ylim(0, None); ax.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("仰角35.264°固定・等間隔方位配置の開始方位依存（□：$A_0$=0°の基準配置）",
                 fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(RES / "fig_azimuth_sweep.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    fig_n3_wait(); fig_azimuth()
