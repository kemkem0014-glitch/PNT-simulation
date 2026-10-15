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


def fig_sky(cache=None):
    """Grayscale sky map of 24-h observation counts for a 10-deg aperture centred at
    each direction (E9_sky_density_N1.csv, all 637 satellites), with the reference
    apertures of N=1..6 overlaid."""
    d = pd.read_csv(RES / "E9_sky_density_N1.csv")
    d = d[d.el < 90]
    A = np.radians(np.append(np.sort(d.az.unique()), 360.0))
    E = np.sort(d.el.unique())
    Z = d.pivot(index="el", columns="az", values="n_obs").values / 1000.0
    Z = np.hstack([Z, Z[:, :1]])
    fig = plt.figure(figsize=(W_IN, W_IN * 0.92))
    axs = [fig.add_axes([0.06 + 0.32 * (k % 3), 0.58 - 0.38 * (k // 3), 0.26, 0.32], projection="polar")
           for k in range(6)]
    from matplotlib.colors import PowerNorm
    th = np.linspace(0, 2 * np.pi, 120)
    for n, ax in zip(range(1, 7), axs):
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        pc = ax.pcolormesh(A, 90 - E, Z, shading="gouraud", cmap="Greys", norm=PowerNorm(0.5, vmin=0, vmax=40), rasterized=True)
        ax.fill_between(np.linspace(0, 2 * np.pi, 200), 80, 90, facecolor="none", edgecolor="#aaaaaa", hatch="////", lw=0)
        ax.set_ylim(0, 90); ax.set_yticks([30, 60]); ax.set_yticklabels([])
        ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels([])
        ax.grid(color="#888888", lw=0.3, ls=":")
        ax.text(0, 103, "N", ha="center", va="bottom", fontsize=6)
        for a, e in base_centers(n):
            z0 = 90 - e
            dz = 5 * np.cos(th); da = np.radians(5 * np.sin(th) / max(math.sin(math.radians(z0)), 0.2))
            ax.fill(np.radians(a) + da, z0 + dz, facecolor="white", edgecolor="black", lw=0.9, zorder=5)
        ax.text(np.radians(-42), 122, f"N={n}", ha="center", va="center", fontsize=7, fontweight="bold")
    cax = fig.add_axes([0.2, 0.12, 0.6, 0.025])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label("開口中心を置いた方向ごとの24時間観測点数 [千点]（斜線：仰角10°未満）", fontsize=6.3)
    cb.ax.tick_params(labelsize=6); cb.set_ticks([0, 5, 10, 20, 30, 40])
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
    a.plot(w[show], d.H_med[show], color="black", lw=1.3, ls="-", label="水平 中央値")
    a.plot(w[show90], d.H_p90[show90], color="black", lw=1.0, ls="--", label="水平 P90")
    a.plot(w[show], d.D_med[show], color="#777777", lw=1.6, ls="-", label="3次元 中央値")
    a.axhline(100, color=INK, lw=0.6, ls=":")
    a.set_yscale("log"); a.set_ylabel("CRLB [m]")
    a.set_yticks([50, 100, 200, 500]); a.set_yticklabels(["50", "100", "200", "500"])
    a.minorticks_off()
    a.grid(True, color=GRID, lw=0.4); a.legend(frameon=False, fontsize=6.5, loc="upper right")
    b = ax[1]
    b.plot(w, d.regular_pct, color="black", lw=1.0, ls=":", label="正則率")
    b.plot(w, d.valid_pct, color="#777777", lw=1.6, ls="-", label="有効率（3D≤1 km）")
    b.plot(w, d.A100, color="black", lw=1.3, ls="-", label="水平≤100 m 達成率")
    b.set_ylim(0, 102); b.set_ylabel("割合 [%]"); b.set_xlabel("待ち時間 T [min]")
    b.grid(True, color=GRID, lw=0.4); b.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3, h_pad=0.4); fig.savefig(OUT / "jin_fig2_wait.png", dpi=300); plt.close(fig)


def fig_az():
    d = pd.read_csv(RES / "E2_azimuth_sweep.csv")
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.55))
    sty = {3: ("o", "-", "black", "black"), 4: ("^", "--", "black", "white"),
           5: ("s", ":", "black", "#999999"), 6: ("D", "-.", "#555555", "white")}
    for n in (3, 4, 5, 6):
        mk, ls, col, face = sty[n]
        s = d[d.N == n].sort_values("A0")
        ax.plot(s.A0, s.H_med, marker=mk, ms=3.2, lw=0.9, ls=ls, color=col, mfc=face, mec="black", mew=0.6, label=f"N={n}")
        ax.plot([0], s.H_med.iloc[:1], marker="o", ms=8, mfc="none", mec="black", mew=0.8, ls="none")
    ax.set_xlabel("開始方位 $A_0$ [deg]")
    ax.set_ylabel("60分 水平CRLB 中央値 [m]")
    ax.set_ylim(0, 62); ax.set_xlim(-3, 120); ax.grid(True, color=GRID, lw=0.4)
    ax.legend(frameon=False, ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "jin_fig3_azimuth.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig_sky(rc.load_cache()); fig_wait(); fig_az()
