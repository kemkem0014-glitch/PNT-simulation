#!/usr/bin/env python3
"""Column-width figures (80 mm) for the JIN-format manuscript, drawn from
results/revision_20261012 and the 1-s geometry cache."""
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import revision_common as rc

RES = rc.ROOT / "results/revision_20261012"
OUT = RES / "jin"
OUT.mkdir(exist_ok=True)
W_IN = 80 / 25.4
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
INK, GRID = "#333333", "#dddddd"
C = ["#2a6fdb", "#e07b39", "#3a9e5f", "#a347b8"]
REP43 = [62880, 62837, 62271, 62883]
REP53 = [59718, 60401, 60070, 61057]


def base_centers(n):
    el = 22.5 if n == 1 else 45.0 if n == 2 else rc.EOPT
    return rc.equal_az(n, 0.0, el)


def fig_sky(cache):
    norad = cache["norad"]
    fig, axs = plt.subplots(2, 3, figsize=(W_IN, W_IN * 0.72), subplot_kw=dict(projection="polar"))
    day = cache["t"] < rc.DAY
    tracks = []
    for nid, ls in [(x, "-") for x in REP43] + [(x, "--") for x in REP53]:
        k = int(np.flatnonzero(norad == nid)[0])
        m = day & (cache["sat"] == k) & (cache["sky"][:, 2] > 0)
        t = cache["t"][m]; s = cache["sky"][m]
        brk = np.flatnonzero(np.diff(t) > 1) + 1
        for seg_t, seg in zip(np.split(t, brk), np.split(s, brk)):
            seg = seg[::30]
            az = np.arctan2(seg[:, 0], seg[:, 1]); zen = 90 - np.degrees(np.arcsin(seg[:, 2]))
            tracks.append((az, zen, ls))
    th = np.linspace(0, 2 * np.pi, 120)
    for n, ax in zip(range(1, 7), axs.flat):
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_ylim(0, 90); ax.set_yticks([30, 60]); ax.set_yticklabels([])
        ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels([])
        ax.text(0, 104, "N", ha="center", va="bottom", fontsize=6)
        ax.grid(color=GRID, lw=0.4)
        for az, zen, ls in tracks:
            ax.plot(az, zen, color="#b0b0b0", lw=0.3, ls=ls)
        for a, e in base_centers(n):
            # 5-deg circle around the centre (small-angle drawing in az/zenith coords)
            z0 = 90 - e
            dz = 5 * np.cos(th); da = np.radians(5 * np.sin(th) / max(math.sin(math.radians(z0)), 0.2))
            ax.fill(np.radians(a) + da, z0 + dz, color=C[0], alpha=0.85, lw=0)
        ax.text(np.radians(-45), 128, f"N={n}", ha="center", va="center", fontsize=7, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.03, wspace=0.3, hspace=0.32)
    fig.savefig(OUT / "jin_fig1_sky.png", dpi=300); plt.close(fig)


def fig_wait():
    d = pd.read_csv(RES / "E0_baseline_all_waits.csv")
    d = d[(d.day == 1) & (d.config == "D3")].sort_values("wait_min")
    w = d.wait_min.to_numpy()
    show = d.valid_pct.to_numpy() >= 50
    show90 = d.valid_pct.to_numpy() >= 90
    fig, ax = plt.subplots(2, 1, figsize=(W_IN, W_IN * 0.78), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.15, 1]))
    a = ax[0]
    a.plot(w[show], d.H_med[show], color=C[0], lw=1.3, label="水平 中央値")
    a.plot(w[show90], d.H_p90[show90], color=C[0], lw=0.9, ls="--", label="水平 P90")
    a.plot(w[show], d.D_med[show], color=C[1], lw=1.3, label="3次元 中央値")
    a.axhline(100, color=INK, lw=0.6, ls=":")
    a.set_yscale("log"); a.set_ylabel("CRLB [m]")
    a.set_yticks([50, 100, 200, 500]); a.set_yticklabels(["50", "100", "200", "500"])
    a.minorticks_off()
    a.grid(True, color=GRID, lw=0.4); a.legend(frameon=False, fontsize=6.5, loc="upper right")
    b = ax[1]
    b.plot(w, d.regular_pct, color=INK, lw=1.0, label="正則率")
    b.plot(w, d.valid_pct, color=C[2], lw=1.3, label="有効率（3D≤1 km）")
    b.plot(w, d.A100, color=C[0], lw=1.3, label="水平≤100 m 達成率")
    b.set_ylim(0, 102); b.set_ylabel("割合 [%]"); b.set_xlabel("待ち時間 T [min]")
    b.grid(True, color=GRID, lw=0.4); b.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3, h_pad=0.4); fig.savefig(OUT / "jin_fig2_wait.png", dpi=300); plt.close(fig)


def fig_az():
    d = pd.read_csv(RES / "E2_azimuth_sweep.csv")
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.55))
    for k, n in enumerate((3, 4, 5, 6)):
        s = d[d.N == n].sort_values("A0")
        ax.plot(s.A0, s.H_med, marker="o", ms=2, lw=1.1, color=C[k], label=f"N={n}")
        ax.plot([0], s.H_med.iloc[:1], marker="s", ms=5, mfc="none", mew=0.8, color=C[k])
    ax.set_xlabel("開始方位 $A_0$ [deg]")
    ax.set_ylabel("60分 水平CRLB 中央値 [m]")
    ax.set_ylim(0, 62); ax.set_xlim(-3, 120); ax.grid(True, color=GRID, lw=0.4)
    ax.legend(frameon=False, ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "jin_fig3_azimuth.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig_sky(rc.load_cache()); fig_wait(); fig_az()
