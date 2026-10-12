#!/usr/bin/env python3
"""Additional experiments for the 2026-10-12 revision (Tables 13-21 of the revised manuscript).

Run:  python research/sky_aperture_leo/revision_experiments_20261012.py
Input: the frozen DTC OMM snapshot used by reproduce_3d_paper_20261006.py.
All outputs go to results/revision_20261012/.  Deterministic except E8, which
uses a fixed random seed.

E0  baseline D1-D6 for every wait 1..60 min on day 1 and day 2
    (validity rate, threshold attainment, T_{h,q})
E1  nested additions at a fixed elevation (two addition orders)
E2  start-azimuth sweep at the LOS-isotropic elevation (5-deg step, full period)
E3  EFIM grid search (elevation x start azimuth), selected on day 1,
    evaluated out of sample on day 2
E4  concentrated vs distributed apertures at the SAME centre elevation
E5  noise scaling (exact)
E6  satellite-wise unknown linear frequency drift
E7  orbit-error nuisance: satellite position AND velocity errors with priors
E8  small Gauss-Newton Monte Carlo (convergence and RMSE / CRLB)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec

import revision_common as rc

OUT = rc.ROOT / "results/revision_20261012"
OUT.mkdir(parents=True, exist_ok=True)
NS = 637
EOPT = rc.EOPT
DAY = rc.DAY
STARTS1 = rc.STARTS
STARTS2 = rc.STARTS + DAY
WAITS7 = [5, 10, 15, 20, 30, 45, 60]
VALID_3D_M = 1000.0    # local-linearisation validity limit for the 3-D CRLB


def base_el(n):
    return 22.5 if n == 1 else 45.0 if n == 2 else EOPT


def sub(cache, mask):
    return cache["t"][mask], cache["sat"][mask]


def run_const(cache, centers, radius=5.0, starts=STARTS1):
    day = (cache["t"] >= starts[0]) & (cache["t"] < starts[-1] + 3600)
    m = rc.gate(cache, centers, radius) & day
    F = rc.efim_constant(cache["t"][m], cache["sat"][m], cache["g"][m], starts, NS)
    h, v, d, lmin, cond = rc.crlb_from_F(F)
    return dict(mask=m, F=F, h=h, v=v, d=d, lmin=lmin, cond=cond,
                n_obs=int(m.sum()), n_sat=int(np.unique(cache["sat"][m]).size))


def med(x):
    """median over starts treating non-regular windows as +inf (failure)."""
    return float(np.median(x))


def p90(x):
    """linear-interpolated 90th percentile; +inf when it falls on a failure."""
    q = float(np.quantile(np.where(np.isfinite(x), x, np.finfo(float).max), 0.9))
    return math.inf if q > 1e300 else q


def summarise(tag, res, waits=WAITS7, extra=None):
    rows = []
    for w in waits:
        i = w - 1
        h, v, d = res["h"][:, i], res["v"][:, i], res["d"][:, i]
        fin = np.isfinite(h)
        valid = fin & (d <= VALID_3D_M)
        row = dict(config=tag, wait_min=w, n_obs=res["n_obs"], n_sat=res["n_sat"],
                   regular_pct=100 * fin.mean(), valid_pct=100 * valid.mean(),
                   H_med=med(h), H_p90=p90(h), V_med=med(v), V_p90=p90(v),
                   D_med=med(d), D_p90=p90(d),
                   H_med_finite=float(np.median(h[fin])) if fin.any() else math.inf,
                   lmin_med=float(np.median(res["lmin"][:, i])),
                   cond_med=float(np.median(res["cond"][:, i])),
                   A100=100 * (h <= 100).mean(), A50=100 * (h <= 50).mean(),
                   A20=100 * (h <= 20).mean())
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def t_hq(res, h_lim, q):
    a = (res["h"] <= h_lim).mean(axis=0)          # attainment per wait 1..60
    ok = np.flatnonzero(a >= q)
    return int(ok[0] + 1) if ok.size else None


# ------------------------------------------------------------------ E0
def e0(cache):
    rows, thq = [], []
    for day, starts in ((1, STARTS1), (2, STARTS2)):
        for n in range(1, 7):
            res = run_const(cache, rc.equal_az(n, 0.0, base_el(n)), starts=starts)
            rows += summarise(f"D{n}", res, waits=range(1, 61), extra=dict(day=day))
            for hl in (100, 50, 20):
                for q in (0.5, 0.9):
                    thq.append(dict(day=day, config=f"D{n}", h_m=hl, q=q,
                                    T_min=t_hq(res, hl, q)))
    pd.DataFrame(rows).to_csv(OUT / "E0_baseline_all_waits.csv", index=False)
    pd.DataFrame(thq).to_csv(OUT / "E0_T_hq.csv", index=False)


# ------------------------------------------------------------------ E1
def e1(cache):
    orders = {"seq": [0, 60, 120, 180, 240, 300], "alt": [0, 180, 60, 240, 120, 300]}
    rows, mono = [], []
    for oname, order in orders.items():
        prev = None
        for k in range(1, 7):
            res = run_const(cache, [(a, EOPT) for a in order[:k]])
            r = summarise(f"{oname}K{k}", res, waits=[60])[0]
            r.update(order=oname, K=k, added_az=order[k - 1])
            rows.append(r)
            if prev is not None:
                inc = np.all(res["h"][:, 59] <= prev[:, 59] * (1 + 1e-9))
                mono.append(dict(order=oname, K=k, all_starts_nonincreasing=bool(inc)))
            prev = res["h"]
    pd.DataFrame(rows).to_csv(OUT / "E1_nested.csv", index=False)
    pd.DataFrame(mono).to_csv(OUT / "E1_nested_monotonicity.csv", index=False)


# ------------------------------------------------------------------ E2
def e2(cache):
    rows = []
    for n in (3, 4, 5, 6):
        for a0 in np.arange(0, 360 / n - 1e-9, 5.0):
            res = run_const(cache, rc.equal_az(n, a0, EOPT))
            r = summarise(f"D{n}", res, waits=[60])[0]
            r.update(N=n, A0=float(a0))
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "E2_azimuth_sweep.csv", index=False)


# ------------------------------------------------------------------ E3
def e3(cache):
    els = [10.0, 15.0, 20.0, 25.0, 30.0, EOPT, 45.0]
    rows = []
    for n in (3, 4, 5, 6):
        for el in els:
            for a0 in np.arange(0, 360 / n - 1e-9, 15.0):
                c = rc.equal_az(n, a0, el)
                r1 = summarise("g", run_const(cache, c, starts=STARTS1), waits=[60])[0]
                r2 = summarise("g", run_const(cache, c, starts=STARTS2), waits=[60])[0]
                rows.append(dict(N=n, El=el, A0=float(a0),
                                 d1_D_med=r1["D_med"], d1_D_p90=r1["D_p90"], d1_H_med=r1["H_med"],
                                 d1_n_obs=r1["n_obs"], d1_n_sat=r1["n_sat"],
                                 d2_D_med=r2["D_med"], d2_D_p90=r2["D_p90"], d2_H_med=r2["H_med"]))
        print("E3 N", n, flush=True)
    # single aperture: elevation x azimuth
    for el in [10.0, 15.0, 20.0, 22.5, 25.0, 30.0, 45.0, 60.0, 90.0]:
        for az in (np.arange(0, 360, 30.0) if el < 90 else [0.0]):
            r1 = summarise("g", run_const(cache, [(az, el)], starts=STARTS1), waits=[60])[0]
            r2 = summarise("g", run_const(cache, [(az, el)], starts=STARTS2), waits=[60])[0]
            rows.append(dict(N=1, El=el, A0=float(az), d1_D_med=r1["D_med"], d1_D_p90=r1["D_p90"],
                             d1_H_med=r1["H_med"], d1_n_obs=r1["n_obs"], d1_n_sat=r1["n_sat"],
                             d2_D_med=r2["D_med"], d2_D_p90=r2["D_p90"], d2_H_med=r2["H_med"]))
    pd.DataFrame(rows).to_csv(OUT / "E3_grid_search.csv", index=False)


# ------------------------------------------------------------------ E4
def e4(cache):
    rows = []
    for n in range(1, 7):
        el = base_el(n)
        alpha = math.degrees(math.acos(1 - n * (1 - math.cos(5 * rc.RAD))))
        big = run_const(cache, [(0.0, el)], radius=alpha)
        dist = run_const(cache, rc.equal_az(n, 0.0, el))
        rb = summarise("C", big, waits=[60])[0]; rd = summarise("D", dist, waits=[60])[0]
        rows.append(dict(N=n, El=el, diameter=2 * alpha,
                         C_obs=rb["n_obs"], C_sat=rb["n_sat"], C_H=rb["H_med"], C_D=rb["D_med"],
                         D_obs=rd["n_obs"], D_sat=rd["n_sat"], D_H=rd["H_med"], D_D=rd["D_med"]))
    pd.DataFrame(rows).to_csv(OUT / "E4_same_elevation_concentrated.csv", index=False)


def e4b(cache):
    """Same-elevation concentrated aperture: sweep its centre azimuth (15-deg step)."""
    rows = []
    for n in range(2, 7):
        el = base_el(n)
        alpha = math.degrees(math.acos(1 - n * (1 - math.cos(5 * rc.RAD))))
        for az in np.arange(0, 360, 15.0):
            r = summarise("C", run_const(cache, [(az, el)], radius=alpha), waits=[60])[0]
            rows.append(dict(N=n, El=el, Az=float(az), C_obs=r["n_obs"], C_sat=r["n_sat"],
                             C_H=r["H_med"], C_D=r["D_med"]))
    pd.DataFrame(rows).to_csv(OUT / "E4b_concentrated_azimuth_sweep.csv", index=False)


# ------------------------------------------------------------------ E6 / E7
def design_rows(cache, m, starts, kind):
    t = cache["t"][m]; g = cache["g"][m]
    if kind == "drift":
        # time in hours relative to the window centre; windows start on STARTS grid,
        # but the bias/drift parameters are per satellite and per window, so the
        # absolute origin does not matter for the Schur complement.
        tau = (t / 3600.0)
        return np.column_stack([g, np.ones(len(t)), tau])
    if kind == "orbit":
        los = cache["los"][m]
        return np.column_stack([g, np.ones(len(t)), -g, los])
    raise ValueError(kind)


def e6_e7(cache):
    rows_d, rows_o = [], []
    for n in range(1, 7):
        c = rc.equal_az(n, 0.0, base_el(n))
        m = rc.gate(cache, c, 5.0) & (cache["t"] < DAY)
        t, s = sub(cache, m)
        # E6 drift (bias + slope flat priors), exact closed form
        F = rc.efim_drift(t, s, cache["g"][m], STARTS1, NS, [30, 60])
        h, v, d, _, _ = rc.crlb_from_F(F)
        for k, w in enumerate((30, 60)):
            rows_d.append(dict(N=n, wait_min=w, model="linear_drift",
                               regular_pct=100 * np.isfinite(h[:, k]).mean(),
                               H_med=med(h[:, k]), H_p90=p90(h[:, k]), D_med=med(d[:, k]),
                               valid_pct=100 * (d[:, k] <= VALID_3D_M).mean()))
        print("E6 N", n, flush=True)
        # E7 orbit error: bias flat, sat position prior tau_p [m], velocity prior tau_v [m/s]
        X = design_rows(cache, m, STARTS1, "orbit")
        for tp, tv in [(10, 0), (100, 0), (1000, 0), (0, 0.01), (0, 0.1), (0, 1.0),
                       (100, 0.1), (1000, 1.0)]:
            cols = [0, 1, 2, 3]; prec = [0.0]
            if tp > 0:
                cols += [4, 5, 6]; prec += [rc.SIGMA ** 2 / tp ** 2] * 3
            if tv > 0:
                cols += [7, 8, 9]; prec += [rc.SIGMA ** 2 / tv ** 2] * 3
            F = rc.efim_general(t, s, X[:, cols], 3, STARTS1, NS, [60], prior_prec=prec)
            h, v, d, _, _ = rc.crlb_from_F(F)
            rows_o.append(dict(N=n, tau_p_m=tp, tau_v_mps=tv, H_med=med(h[:, 0]),
                               H_p90=p90(h[:, 0]), D_med=med(d[:, 0])))
        print("E7 N", n, flush=True)
    pd.DataFrame(rows_d).to_csv(OUT / "E6_linear_drift.csv", index=False)
    pd.DataFrame(rows_o).to_csv(OUT / "E7_orbit_error.csv", index=False)


# ------------------------------------------------------------------ E8
def e8(cache, seed=20261012):
    """Gauss-Newton on [r, beta_i] with beta eliminated by centering."""
    rng = np.random.default_rng(seed)
    with rc.rp.INFILE.open(encoding="utf-8") as f:
        fields = [x for x in omm.parse_csv(f) if "[DTC]" in x.get("OBJECT_NAME", "")]
    sats = []
    for f in fields:
        o = Satrec(); omm.initialize(o, f); sats.append(o)
    r_true = rc.rp.R_OBS * 1000.0                      # m
    rows = []
    for n in (3, 6):
        c = rc.equal_az(n, 0.0, EOPT)
        mall = rc.gate(cache, c, 5.0) & (cache["t"] < DAY)
        for s0 in STARTS1[::69][:10]:
            m = mall & (cache["t"] >= s0) & (cache["t"] < s0 + 3600)
            t = cache["t"][m]; sid = cache["sat"][m].astype(int)
            P = np.empty((len(t), 3)); V = np.empty((len(t), 3))
            for k in np.unique(sid):
                sel = sid == k
                j0, jf, jj = rc.rp.julian(rc.rp.T0 + t[sel])
                _, pr, pv = sats[k].sgp4_array(j0, jf)
                pp, vv = rc.rp.ecef(pr, pv, jj)
                P[sel] = pp * 1000; V[sel] = vv * 1000      # m, m/s

            def model(r):
                d = P - r; rho = np.linalg.norm(d, axis=1); e = d / rho[:, None]
                rr = np.einsum("ij,ij->i", e, V)
                g = -(V - e * rr[:, None]) / rho[:, None]
                return rr, g

            y0, g0 = model(r_true)
            keys, inv = np.unique(sid, return_inverse=True)

            def center(x):
                cnt = np.bincount(inv)
                mu = np.stack([np.bincount(inv, weights=x[:, j]) / cnt for j in range(x.shape[1])], 1) \
                    if x.ndim == 2 else np.bincount(inv, weights=x) / cnt
                return x - mu[inv]

            gc = center(g0)
            J = gc.T @ gc / rc.SIGMA ** 2
            crlb3d = math.sqrt(np.trace(np.linalg.inv(J)))
            beta = rng.normal(0, 500.0, size=len(keys))       # arbitrary large biases (m/s)
            for trial in range(20):
                y = y0 + beta[inv] + rng.normal(0, rc.SIGMA, size=len(t))
                for off_km in (1.0, 10.0, 100.0):
                    u = rng.normal(size=3); u /= np.linalg.norm(u)
                    r = r_true + u * off_km * 1000
                    ok = False
                    for it in range(30):
                        yh, g = model(r)
                        res = center(y - yh); gcc = center(g)
                        dr, *_ = np.linalg.lstsq(gcc, res, rcond=None)
                        r = r + dr
                        if np.linalg.norm(dr) < 1e-3:
                            ok = True; break
                    yh, _ = model(r)
                    rms_res = float(np.sqrt(np.mean(center(y - yh) ** 2)))
                    rows.append(dict(N=n, start=int(s0), trial=trial, init_offset_km=off_km,
                                     converged=ok, iters=it + 1,
                                     err_3d_m=float(np.linalg.norm(r - r_true)),
                                     crlb_3d_m=crlb3d, rms_resid_mps=rms_res))
        print("E8 N", n, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "E8_gauss_newton_mc.csv", index=False)


def main():
    t0 = time.time()
    cache = rc.load_cache()
    meta = dict(script="revision_experiments_20261012.py", source_sha256=__import__("hashlib")
                .sha256(rc.rp.INFILE.read_bytes()).hexdigest(),
                cache_samples=int(len(cache["t"])), days_cached=2,
                valid_3d_limit_m=VALID_3D_M, sigma_rr_mps=rc.SIGMA,
                rank_rule="eig_min>1e-18 and eig_min/eig_max>1e-12")
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for name, fn in [("E0", e0), ("E1", e1), ("E2", e2), ("E4", e4), ("E4b", e4b), ("E6/E7", e6_e7),
                     ("E8", e8), ("E3", e3)]:
        fn(cache); print("done", name, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
