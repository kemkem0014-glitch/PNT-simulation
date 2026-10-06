#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sgp4 import omm
from sgp4.api import Satrec

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv"
OMEGA_EARTH = 7.2921150e-5
WGS84_A_KM = 6378.137
WGS84_E2 = 6.69437999014e-3

RTK_LAT_DEG = 35.6662430056
RTK_LON_DEG = 139.7923087000
RTK_H_M = 59.9

APERTURES = (
    ("A1_NE", 45.0, 45.0),
    ("A2_S", 180.0, 55.0),
    ("A3_NW", 315.0, 45.0),
)
APERTURE_DIAMETER_DEG = 10.0
SIGMA_RR_MPS = 0.2


def parse_utc(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def observer_ecef_and_basis(lat_deg, lon_deg, h_m):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    a = WGS84_A_KM
    N = a / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    h = h_m / 1000.0
    r = np.array([
        (N + h) * math.cos(lat) * math.cos(lon),
        (N + h) * math.cos(lat) * math.sin(lon),
        (N * (1.0 - WGS84_E2) + h) * math.sin(lat),
    ])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([
        -math.sin(lat) * math.cos(lon),
        -math.sin(lat) * math.sin(lon),
        math.cos(lat),
    ])
    up = np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ])
    return r, east, north, up


def aperture_unit(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array([
        math.cos(el) * math.sin(az),
        math.cos(el) * math.cos(az),
        math.sin(el),
    ])


def gmst_rad(jd):
    T = (jd - 2451545.0) / 36525.0
    sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    return np.deg2rad((sec / 240.0) % 360.0)


def teme_to_ecef(r_teme, v_teme, jd):
    th = gmst_rad(jd)
    c = np.cos(th)
    s = np.sin(th)
    x = c * r_teme[:, 0] + s * r_teme[:, 1]
    y = -s * r_teme[:, 0] + c * r_teme[:, 1]
    z = r_teme[:, 2]
    r = np.column_stack((x, y, z))

    vx0 = c * v_teme[:, 0] + s * v_teme[:, 1]
    vy0 = -s * v_teme[:, 0] + c * v_teme[:, 1]
    vz0 = v_teme[:, 2]
    v = np.column_stack((
        vx0 + OMEGA_EARTH * y,
        vy0 - OMEGA_EARTH * x,
        vz0,
    ))
    return r, v


def unix_to_jd(seconds):
    jd = np.asarray(seconds, dtype=float) / 86400.0 + 2440587.5
    jdi = np.floor(jd)
    fr = jd - jdi
    return jdi, fr, jd


def geometry(r_ecef, v_ecef, r_obs, east, north, up):
    d = r_ecef - r_obs
    rho = np.linalg.norm(d, axis=1)
    u = d / rho[:, None]
    enu = np.column_stack((u @ east, u @ north, u @ up))

    uv = np.sum(u * v_ecef, axis=1)
    perp = v_ecef - u * uv[:, None]
    grad_ecef = -perp / rho[:, None]
    g_e = grad_ecef @ east
    g_n = grad_ecef @ north
    return enu, g_e, g_n


def fetch_dtc(url):
    response = requests.get(url, timeout=60, headers={"User-Agent": "PNT-simulation research"})
    response.raise_for_status()
    text = response.text
    all_fields = list(omm.parse_csv(io.StringIO(text)))
    dtc = [f for f in all_fields if "[DTC]" in f.get("OBJECT_NAME", "")]
    if len(dtc) < 300:
        raise RuntimeError(
            f"Only {len(dtc)} [DTC]-tagged Starlink records found; expected hundreds. "
            "Refusing to silently substitute shell-synthetic satellites."
        )
    return text, dtc


def build_satellites(fields):
    out = []
    for f in fields:
        sat = Satrec()
        omm.initialize(sat, f)
        out.append((f, sat))
    return out


def simulate_observations(satellites, start_dt, hours, coarse_step_s):
    start_unix = start_dt.timestamp()
    stop_unix = start_unix + hours * 3600.0
    coarse = np.arange(start_unix, stop_unix + 0.1, coarse_step_s)
    jd_i, fr, jd_full = unix_to_jd(coarse)

    r_obs, east, north, up = observer_ecef_and_basis(
        RTK_LAT_DEG, RTK_LON_DEG, RTK_H_M
    )
    aperture_vecs = {name: aperture_unit(az, el) for name, az, el in APERTURES}
    radius = APERTURE_DIAMETER_DEG / 2.0
    candidate_radius = radius + 4.0

    rows = []
    skipped = 0
    for idx, (fields, sat) in enumerate(satellites, 1):
        err, r_teme, v_teme = sat.sgp4_array(jd_i, fr)
        ok = err == 0
        if not np.any(ok):
            skipped += 1
            continue
        r_ecef, v_ecef = teme_to_ecef(r_teme, v_teme, jd_full)
        enu, _, _ = geometry(r_ecef, v_ecef, r_obs, east, north, up)

        cand = np.zeros(len(coarse), dtype=bool)
        for avec in aperture_vecs.values():
            dot = np.clip(enu @ avec, -1.0, 1.0)
            sep = np.degrees(np.arccos(dot))
            cand |= sep <= candidate_radius
        cand &= ok
        ci = np.flatnonzero(cand)
        if len(ci) == 0:
            continue

        refine_seconds = set()
        pad = int(math.ceil(coarse_step_s + 5))
        for i in ci:
            center = int(round(coarse[i]))
            refine_seconds.update(range(center - pad, center + pad + 1))
        refine = np.array(sorted(t for t in refine_seconds if start_unix <= t < stop_unix), dtype=float)
        rjdi, rfr, rjd = unix_to_jd(refine)
        e2, rr_teme, vv_teme = sat.sgp4_array(rjdi, rfr)
        valid = e2 == 0
        if not np.any(valid):
            continue
        rr_ecef, vv_ecef = teme_to_ecef(rr_teme, vv_teme, rjd)
        renu, g_e, g_n = geometry(rr_ecef, vv_ecef, r_obs, east, north, up)

        for ap_name, avec in aperture_vecs.items():
            dot = np.clip(renu @ avec, -1.0, 1.0)
            sep = np.degrees(np.arccos(dot))
            hit = valid & (sep <= radius)
            for j in np.flatnonzero(hit):
                rows.append((
                    refine[j] - start_unix,
                    int(fields["NORAD_CAT_ID"]),
                    fields["OBJECT_NAME"],
                    float(fields["INCLINATION"]),
                    ap_name,
                    float(sep[j]),
                    float(g_e[j]),
                    float(g_n[j]),
                ))
        if idx % 50 == 0:
            print(f"propagated {idx}/{len(satellites)} DTC satellites; observations={len(rows)}", flush=True)

    obs = pd.DataFrame(rows, columns=[
        "t_s", "norad_id", "name", "inclination_deg", "aperture",
        "separation_deg", "grad_e_sinv", "grad_n_sinv",
    ])
    if len(obs):
        obs = obs.drop_duplicates(["t_s", "norad_id", "aperture"]).sort_values("t_s").reset_index(drop=True)
    print(f"finished propagation: {len(satellites)} DTC satellites, {len(obs)} 1-Hz aperture observations, skipped={skipped}")
    return obs


def position_crlb(window):
    if window.empty:
        return math.inf, math.nan, 0
    J = np.zeros((2, 2), float)
    used = 0
    for _, grp in window.groupby("norad_id"):
        G = grp[["grad_e_sinv", "grad_n_sinv"]].to_numpy(float)
        if len(G) < 2:
            continue
        centered = G - G.mean(axis=0, keepdims=True)
        Ji = centered.T @ centered / (SIGMA_RR_MPS ** 2)
        if np.trace(Ji) > 0:
            J += Ji
            used += 1
    eig = np.linalg.eigvalsh(J)
    if used == 0 or eig[0] <= 1e-18:
        return math.inf, float(eig[0]) if len(eig) else math.nan, used
    P = np.linalg.inv(J)
    return float(math.sqrt(np.trace(P))), float(eig[0]), used


def occupancy_pattern(window):
    counts = []
    for name, _, _ in APERTURES:
        counts.append(int(window.loc[window["aperture"] == name, "norad_id"].nunique()))
    return "+".join(str(x) for x in sorted(counts, reverse=True)), counts


def evaluate_waiting(obs, hours, start_step_min=2, max_wait_min=60):
    end_s = hours * 3600.0
    starts = np.arange(0.0, end_s - max_wait_min * 60.0 + 0.1, start_step_min * 60.0)
    rows = []
    for si, s0 in enumerate(starts):
        for wait in range(1, max_wait_min + 1):
            w = obs[(obs["t_s"] >= s0) & (obs["t_s"] < s0 + wait * 60.0)]
            hstd, lam_min, used_bias_elim = position_crlb(w)
            aps = int(w["aperture"].nunique()) if len(w) else 0
            nsat = int(w["norad_id"].nunique()) if len(w) else 0
            patt, counts = occupancy_pattern(w) if len(w) else ("0+0+0", [0, 0, 0])
            rows.append((
                int(round(s0)), wait, len(w), nsat, aps, patt,
                counts[0], counts[1], counts[2], used_bias_elim,
                hstd, lam_min,
            ))
        if (si + 1) % 100 == 0:
            print(f"waiting statistics {si+1}/{len(starts)} starts", flush=True)
    return pd.DataFrame(rows, columns=[
        "start_offset_s", "wait_min", "n_observations", "n_satellites",
        "n_apertures_used", "occupancy_pattern",
        "n_sat_ap1", "n_sat_ap2", "n_sat_ap3", "n_satellites_with_2plus_obs",
        "horizontal_crlb_m", "lambda_min",
    ])


def summarize(waitdf):
    summary = []
    for wait, g in waitdf.groupby("wait_min"):
        finite = g[np.isfinite(g["horizontal_crlb_m"])]
        all3 = g[g["n_apertures_used"] == 3]
        all3f = all3[np.isfinite(all3["horizontal_crlb_m"])]
        row = {
            "wait_min": int(wait),
            "n_start_times": len(g),
            "all3_apertures_percent": 100.0 * len(all3) / len(g),
            "finite_solution_percent": 100.0 * len(finite) / len(g),
            "all3_finite_percent": 100.0 * len(all3f) / len(g),
            "median_satellites_all": float(g["n_satellites"].median()),
            "median_satellites_all3": float(all3["n_satellites"].median()) if len(all3) else math.nan,
            "median_crlb_all_m": float(finite["horizontal_crlb_m"].median()) if len(finite) else math.nan,
            "p90_crlb_all_m": float(finite["horizontal_crlb_m"].quantile(.90)) if len(finite) else math.nan,
            "median_crlb_all3_m": float(all3f["horizontal_crlb_m"].median()) if len(all3f) else math.nan,
            "p90_crlb_all3_m": float(all3f["horizontal_crlb_m"].quantile(.90)) if len(all3f) else math.nan,
        }
        for target in (500, 200, 100, 50):
            row[f"pct_all3_crlb_le_{target}m"] = 100.0 * np.mean(
                (g["n_apertures_used"] == 3) &
                np.isfinite(g["horizontal_crlb_m"]) &
                (g["horizontal_crlb_m"] <= target)
            )
        summary.append(row)
    return pd.DataFrame(summary)


def satellite_count_summary(waitdf):
    selected = waitdf[waitdf["wait_min"].isin([5, 10, 15, 20, 30, 45, 60])]
    selected = selected[(selected["n_apertures_used"] == 3) & np.isfinite(selected["horizontal_crlb_m"])]
    rows = []
    for (wait, nsat), g in selected.groupby(["wait_min", "n_satellites"]):
        rows.append({
            "wait_min": int(wait),
            "n_satellites": int(nsat),
            "cases": len(g),
            "median_crlb_m": float(g["horizontal_crlb_m"].median()),
            "p25_crlb_m": float(g["horizontal_crlb_m"].quantile(.25)),
            "p75_crlb_m": float(g["horizontal_crlb_m"].quantile(.75)),
            "p90_crlb_m": float(g["horizontal_crlb_m"].quantile(.90)),
        })
    return pd.DataFrame(rows)


def occupancy_summary(waitdf):
    selected = waitdf[waitdf["wait_min"].isin([10, 20, 30, 60])]
    selected = selected[(selected["n_apertures_used"] == 3) & np.isfinite(selected["horizontal_crlb_m"])]
    rows = []
    for (wait, patt), g in selected.groupby(["wait_min", "occupancy_pattern"]):
        rows.append({
            "wait_min": int(wait),
            "occupancy_pattern": patt,
            "cases": len(g),
            "median_satellites": float(g["n_satellites"].median()),
            "median_crlb_m": float(g["horizontal_crlb_m"].median()),
            "p90_crlb_m": float(g["horizontal_crlb_m"].quantile(.90)),
        })
    return pd.DataFrame(rows).sort_values(["wait_min", "cases"], ascending=[True, False])


def attainment(waitdf):
    rows = []
    for s0, g in waitdf.groupby("start_offset_s"):
        g = g.sort_values("wait_min")
        for target in (500, 200, 100, 50):
            good = g[
                (g["n_apertures_used"] == 3)
                & np.isfinite(g["horizontal_crlb_m"])
                & (g["horizontal_crlb_m"] <= target)
            ]
            rows.append({
                "start_offset_s": int(s0),
                "target_m": target,
                "first_wait_min": int(good["wait_min"].iloc[0]) if len(good) else math.nan,
            })
    raw = pd.DataFrame(rows)
    out = []
    for target, g in raw.groupby("target_m"):
        ok = g["first_wait_min"].dropna()
        out.append({
            "target_m": int(target),
            "success_within_60min_percent": 100.0 * len(ok) / len(g),
            "median_wait_min_if_success": float(ok.median()) if len(ok) else math.nan,
            "p90_wait_min_if_success": float(ok.quantile(.90)) if len(ok) else math.nan,
        })
    return raw, pd.DataFrame(out)


def make_figures(summary, sat_summary, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["wait_min"], summary["median_crlb_all3_m"], label="Median, 3 apertures")
    ax.plot(summary["wait_min"], summary["p90_crlb_all3_m"], label="P90, 3 apertures")
    ax.set_yscale("log")
    ax.set_xlabel("Waiting time (min)")
    ax.set_ylabel("Horizontal CRLB 1-sigma (m)")
    ax.set_title("TUMSAT RTK point: waiting time vs DTC Doppler geometry")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "wait_vs_crlb_3ap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for target in (500, 200, 100, 50):
        ax.plot(summary["wait_min"], summary[f"pct_all3_crlb_le_{target}m"], label=f"<= {target} m")
    ax.set_xlabel("Waiting time (min)")
    ax.set_ylabel("Probability over start times (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Probability of reaching target accuracy with all 3 apertures used")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "target_probability_by_wait.png", dpi=180)
    plt.close(fig)

    if len(sat_summary):
        s60 = sat_summary[sat_summary["wait_min"] == 60]
        if len(s60):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(s60["n_satellites"], s60["median_crlb_m"], marker="o")
            ax.set_yscale("log")
            ax.set_xlabel("Unique DTC satellites observed in 60 min")
            ax.set_ylabel("Median horizontal CRLB 1-sigma (m)")
            ax.set_title("Three fixed 10-deg apertures: satellite diversity at 60 min")
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(outdir / "satellites_vs_crlb_60min.png", dpi=180)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-06T00:00:00Z")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--coarse-step-s", type=float, default=5.0)
    ap.add_argument("--start-step-min", type=int, default=2)
    ap.add_argument("--max-wait-min", type=int, default=60)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Fetching current CelesTrak Starlink OMM CSV...", flush=True)
    raw, fields = fetch_dtc(CELESTRAK_URL)
    sats = build_satellites(fields)
    start = parse_utc(args.start)

    dtc_df = pd.DataFrame(fields)
    dtc_df.to_csv(out / "dtc_omm_snapshot.csv", index=False)
    shell_counts = (
        dtc_df.assign(shell=np.where(
            np.abs(dtc_df["INCLINATION"].astype(float) - 43.0) <
            np.abs(dtc_df["INCLINATION"].astype(float) - 53.0),
            "43deg", "53deg"
        ))
        .groupby("shell").size().rename("count").reset_index()
    )
    shell_counts.to_csv(out / "dtc_shell_counts.csv", index=False)

    obs = simulate_observations(sats, start, args.hours, args.coarse_step_s)
    obs.to_csv(out / "aperture_observations_1hz.csv", index=False)

    waitdf = evaluate_waiting(obs, args.hours, args.start_step_min, args.max_wait_min)
    waitdf.to_csv(out / "waiting_cases.csv", index=False)

    summary = summarize(waitdf)
    summary.to_csv(out / "waiting_summary.csv", index=False)

    satsum = satellite_count_summary(waitdf)
    satsum.to_csv(out / "three_aperture_satellite_count_summary.csv", index=False)

    occsum = occupancy_summary(waitdf)
    occsum.to_csv(out / "three_aperture_occupancy_summary.csv", index=False)

    attr_raw, attr_summary = attainment(waitdf)
    attr_raw.to_csv(out / "threshold_attainment_cases.csv", index=False)
    attr_summary.to_csv(out / "threshold_attainment_summary.csv", index=False)

    make_figures(summary, satsum, out)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "celestrak_url": CELESTRAK_URL,
        "dtc_filter": "OBJECT_NAME contains [DTC]",
        "dtc_satellite_count": len(fields),
        "simulation_start_utc": start.isoformat(),
        "simulation_hours": args.hours,
        "coarse_detection_step_s": args.coarse_step_s,
        "fine_observation_step_s": 1,
        "waiting_start_step_min": args.start_step_min,
        "max_wait_min": args.max_wait_min,
        "reference_point": {
            "name": "Tokyo University of Marine Science and Technology, Kubo Lab public RTK base",
            "lat_deg": RTK_LAT_DEG,
            "lon_deg": RTK_LON_DEG,
            "height_m": RTK_H_M,
        },
        "apertures": [
            {"name": n, "az_deg": az, "el_deg": el, "diameter_deg": APERTURE_DIAMETER_DEG}
            for n, az, el in APERTURES
        ],
        "measurement_model": {
            "observable": "equivalent geometric range-rate / Doppler",
            "sigma_range_rate_mps": SIGMA_RR_MPS,
            "nuisance": "independent constant frequency/range-rate bias per satellite",
            "crlb": "2-D horizontal information after Schur elimination of per-satellite biases",
        },
        "caveat": (
            "This is a current-element-set geometry study, not a statement that every catalogued DTC "
            "satellite is transmitting a usable positioning observable in Japan. SGP4 propagation from "
            "a single snapshot is intentionally limited to the configured short horizon."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDTC shell counts")
    print(shell_counts.to_string(index=False))
    print("\nSelected waiting-time summary")
    print(summary[summary["wait_min"].isin([5,10,15,20,30,45,60])].to_string(index=False))
    print("\nThreshold attainment")
    print(attr_summary.to_string(index=False))


if __name__ == "__main__":
    main()
