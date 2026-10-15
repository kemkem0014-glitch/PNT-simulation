#!/usr/bin/env python3
"""図1（天空開口の基準配置と、開口位置ごとの24時間観測点数）を作るスタンドアロンのスクリプト。

入力 : results/tumsat_real_dtc_20261006/dtc_omm_snapshot.csv（CelesTrak GP/OMM、2026-10-06保存）
出力 : results/revision_20261012/jin/fig1_sky_density.png（図1）、同 fig1_sky_density.csv（濃淡の元データ）
必要 : pip install numpy pandas matplotlib sgp4
実行 : python make_fig1.py

処理の流れ
  1. OMMファイルから名称に[DTC]を含む637機を読み込む
  2. SGP4で2026-10-06 00:00 UTCから24時間、各衛星の位置を計算する
     （60秒刻みで仰角5°以上の区間を探し、前後180秒を加えた区間を1秒刻みで再計算）
  3. TEME座標をGMST回転でECEFへ変換し、越中島の受信点から見た方位・仰角を求める
  4. 方位5°・仰角2.5°刻みの各方向に直径10°の開口を置いたとき、
     24時間に開口内（中心から5°以内）に入る1秒ごとの観測点の数を数える
  5. その数を灰色の濃淡で描き、N=1～6の基準配置の開口を白丸で重ねる
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from sgp4 import omm
from sgp4.api import Satrec

HERE = Path(__file__).resolve().parent
OMM_FILE = HERE / "results/tumsat_real_dtc_20261006/dtc_omm_snapshot.csv"
OUT_PNG = HERE / "results/revision_20261012/jin/fig1_sky_density.png"
OUT_CSV = HERE / "results/revision_20261012/jin/fig1_sky_density.csv"

T0_UNIX = 1791244800          # 2026-10-06 00:00:00 UTC
DAY = 86400
LAT, LON, H_KM = 35.6662430056, 139.7923087, 0.0599   # 越中島RTK基準点（楕円体高59.9 m）
A_WGS, E2_WGS = 6378.137, 6.69437999014e-3
RAD = math.pi / 180
APERTURE_RADIUS = 5.0         # 開口の角半径 [deg]（直径10°）
E_ISO = math.degrees(math.atan(1 / math.sqrt(2)))   # 35.264°


# ---------------------------------------------------------------- 受信点とENU基底
def receiver():
    lat, lon = LAT * RAD, LON * RAD
    n = A_WGS / math.sqrt(1 - E2_WGS * math.sin(lat) ** 2)
    r = np.array([(n + H_KM) * math.cos(lat) * math.cos(lon),
                  (n + H_KM) * math.cos(lat) * math.sin(lon),
                  (n * (1 - E2_WGS) + H_KM) * math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    return r, np.stack((east, north, up))


R_RX, ENU = receiver()


def julian(seconds):
    jd = np.asarray(seconds, float) / 86400 + 2440587.5
    j0 = np.floor(jd)
    return j0, jd - j0, jd


def teme_to_ecef(r, jd):
    """GMST（グリニッジ平均恒星時）で地球の自転角だけ回転させる。"""
    d = (jd - 2451545.0) / 36525
    sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * d + 0.093104 * d * d - 6.2e-6 * d ** 3
    th = np.deg2rad((sec / 240) % 360)
    c, s = np.cos(th), np.sin(th)
    return np.stack((c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1], r[:, 2]), axis=1)


def sky_vectors(r_ecef):
    """受信点から見た視線方向の単位ベクトル（東・北・上の成分）。"""
    d = r_ecef - R_RX
    los = d / np.linalg.norm(d, axis=1)[:, None]
    return los @ ENU.T


def unit(az, el):
    az, el = az * RAD, el * RAD
    return np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


# ---------------------------------------------------------------- 1. 2. 3. 軌道計算
def visible_samples():
    with OMM_FILE.open(encoding="utf-8") as f:
        fields = [x for x in omm.parse_csv(f) if "[DTC]" in x.get("OBJECT_NAME", "")]
    print("DTC衛星数:", len(fields))
    coarse = np.arange(0, DAY + 1, 60)
    c0, cf, cj = julian(T0_UNIX + coarse)
    sky_all = []
    for k, f in enumerate(fields):
        sat = Satrec(); omm.initialize(sat, f)
        err, r, _ = sat.sgp4_array(c0, cf)
        sky = sky_vectors(teme_to_ecef(r, cj))
        vis = (err == 0) & (sky[:, 2] > math.sin(5 * RAD))
        if not vis.any():
            continue
        idx = np.flatnonzero(vis)
        segs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        fine = np.unique(np.concatenate([np.arange(max(0, coarse[s[0]] - 180), min(DAY - 1, coarse[s[-1]] + 180) + 1)
                                         for s in segs]))
        f0, ff, fj = julian(T0_UNIX + fine)
        err, r, _ = sat.sgp4_array(f0, ff)
        sky = sky_vectors(teme_to_ecef(r, fj))
        keep = (err == 0) & (sky[:, 2] > math.sin(5 * RAD))
        sky_all.append(sky[keep].astype(np.float32))
        if (k + 1) % 100 == 0:
            print("  計算済み", k + 1)
    return np.concatenate(sky_all)


# ---------------------------------------------------------------- 4. 開口位置ごとの観測点数
def density(sky):
    cos_r = math.cos(math.radians(APERTURE_RADIUS))
    rows = []
    for el in np.arange(10, 90.1, 2.5):
        azs = np.arange(0, 360, 5.0) if el < 90 else [0.0]
        U = np.stack([unit(a, el) for a in azs]).astype(np.float32)
        counts = ((sky @ U.T) >= cos_r).sum(axis=0)
        rows += [(float(a), float(el), int(n)) for a, n in zip(azs, counts)]
    return pd.DataFrame(rows, columns=["az", "el", "n_obs"])


# ---------------------------------------------------------------- 5. 作図
def base_centers(n):
    el = 22.5 if n == 1 else 45.0 if n == 2 else E_ISO
    return [(360.0 * j / n, el) for j in range(n)]


def plot(d):
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7.5})
    d = d[d.el < 90]
    A = np.radians(np.append(np.sort(d.az.unique()), 360.0))
    E = np.sort(d.el.unique())
    Z = d.pivot(index="el", columns="az", values="n_obs").values / 1000.0
    Z = np.hstack([Z, Z[:, :1]])
    w = 80 / 25.4
    fig = plt.figure(figsize=(w, w * 0.92))
    th = np.linspace(0, 2 * np.pi, 120)
    for k, n in enumerate(range(1, 7)):
        ax = fig.add_axes([0.06 + 0.32 * (k % 3), 0.58 - 0.38 * (k // 3), 0.26, 0.32], projection="polar")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        pc = ax.pcolormesh(A, 90 - E, Z, shading="gouraud", cmap="Greys",
                           norm=PowerNorm(0.5, vmin=0, vmax=40), rasterized=True)
        ax.fill_between(np.linspace(0, 2 * np.pi, 200), 80, 90, facecolor="none",
                        edgecolor="#aaaaaa", hatch="////", lw=0)          # 仰角10°未満は未計算
        ax.set_ylim(0, 90); ax.set_yticks([30, 60]); ax.set_yticklabels([])
        ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels([])
        ax.grid(color="#888888", lw=0.3, ls=":")
        ax.text(0, 103, "N", ha="center", va="bottom", fontsize=6)
        for a, e in base_centers(n):                                      # 開口（半径5°）を白丸で描く
            z0 = 90 - e
            ax.fill(np.radians(a) + np.radians(5 * np.sin(th) / max(math.sin(math.radians(z0)), 0.2)),
                    z0 + 5 * np.cos(th), facecolor="white", edgecolor="black", lw=0.9, zorder=5)
        ax.text(np.radians(-42), 122, f"N={n}", ha="center", va="center", fontsize=7, fontweight="bold")
    cax = fig.add_axes([0.2, 0.12, 0.6, 0.025])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label("開口中心を置いた方向ごとの24時間観測点数 [千点]（斜線：仰角10°未満）", fontsize=6.3)
    cb.ax.tick_params(labelsize=6); cb.set_ticks([0, 5, 10, 20, 30, 40])
    fig.savefig(OUT_PNG, dpi=300); plt.close(fig)


if __name__ == "__main__":
    sky = visible_samples()
    print("仰角5°以上の1秒サンプル数:", len(sky))
    d = density(sky)
    d.to_csv(OUT_CSV, index=False)
    print("最多の方向:\n", d.sort_values("n_obs", ascending=False).head(3).to_string(index=False))
    plot(d)
    print("出力:", OUT_PNG)
