#!/usr/bin/env python3
"""Shared geometry cache and EFIM evaluators for the 2026-10-12 revision experiments.

Re-uses the frozen input, receiver, SGP4 / TEME->ECEF conventions and the
1-s candidate-interval rule of reproduce_3d_paper_20261006.py, but stores every
1-s sample above 5 deg (with the same +/-180 s padding) once, so that any
aperture configuration can be evaluated by a dot-product gate.

Cache columns (per 1-s sample):
  t        seconds after 2026-10-06 00:00 UTC (two days are cached)
  sat      satellite index (0..636, order of the frozen OMM file)
  sky      ENU line-of-sight unit vector (E,N,U)
  g        ECEF range-rate gradient w.r.t. receiver position [1/s]
  los      ECEF LOS unit vector (receiver -> satellite)
  rho      range [km]
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
from sgp4 import omm
from sgp4.api import Satrec

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rp", ROOT / "reproduce_3d_paper_20261006.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

SIGMA = rp.SIGMA
RAD = rp.RAD
EOPT = rp.EOPT
ROT = rp.ROT
STARTS = rp.STARTS            # 691 starts, 0 .. 82800 s, 120 s step
DAY = rp.DAY
CACHE = ROOT / "results/revision_20261012/geometry_cache_2days.npz"


def build_cache(days=2):
    with rp.INFILE.open(encoding="utf-8") as f:
        fields = [x for x in omm.parse_csv(f) if "[DTC]" in x.get("OBJECT_NAME", "")]
    assert len(fields) == rp.NSAT
    span = days * DAY
    coarse = np.arange(0, span + 1, 60, dtype=np.int64)
    j0, jf, j = rp.julian(rp.T0 + coarse)
    T, S, SKY, G, LOS, RHO = [], [], [], [], [], []
    norad = np.zeros(len(fields), dtype=np.int64)
    for k, f in enumerate(fields):
        sat = Satrec(); omm.initialize(sat, f)
        norad[k] = int(f["NORAD_CAT_ID"])
        err, rr, vv = sat.sgp4_array(j0, jf)
        ok = err == 0
        if not ok.any():
            continue
        r, v = rp.ecef(rr, vv, j)
        sky, _, _, _ = rp.geom(r, v)
        det = ok & (sky[:, 2] > math.sin(5 * RAD))
        if not det.any():
            continue
        times = []
        for sp in rp.segments(np.flatnonzero(det)):
            lo = max(0, int(coarse[sp[0]]) - 180)
            hi = min(span - 1, int(coarse[sp[-1]]) + 180)
            times.append(np.arange(lo, hi + 1, dtype=np.int64))
        fine = np.unique(np.concatenate(times))
        a0, af, aj = rp.julian(rp.T0 + fine)
        e2, pr, pv = sat.sgp4_array(a0, af)
        valid = e2 == 0
        pp, vp = rp.ecef(pr, pv, aj)
        sky, grad, rho, _ = rp.geom(pp, vp)
        los = (pp - rp.R_OBS) / rho[:, None]
        keep = valid & (sky[:, 2] > 0)
        T.append(fine[keep]); S.append(np.full(keep.sum(), k, dtype=np.int16))
        SKY.append(sky[keep]); G.append(grad[keep]); LOS.append(los[keep]); RHO.append(rho[keep])
        if (k + 1) % 100 == 0:
            print("cache", k + 1, flush=True)
    order = np.argsort(np.concatenate(T), kind="stable")
    data = dict(t=np.concatenate(T)[order], sat=np.concatenate(S)[order],
                sky=np.concatenate(SKY)[order], g=np.concatenate(G)[order],
                los=np.concatenate(LOS)[order], rho=np.concatenate(RHO)[order], norad=norad)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **data)
    return data


def load_cache():
    if not CACHE.exists():
        return build_cache()
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


def sky_unit(az, el):
    return rp.sky_unit(az, el)


def gate(cache, centers, radius_deg):
    """Boolean mask of cached samples inside ANY circular aperture."""
    U = np.stack([sky_unit(a, e) for a, e in centers])
    return (cache["sky"] @ U.T >= math.cos(radius_deg * RAD)).any(axis=1)


def equal_az(n, a0, el):
    return [((a0 + 360.0 * j / n) % 360.0, el) for j in range(n)]


# ---------------------------------------------------------------- evaluators
def _window_moments(t, sat, cols, start, ns):
    """Cumulative per-minute, per-satellite sums of the given column arrays."""
    lo = np.searchsorted(t, start, "left"); hi = np.searchsorted(t, start + 3600, "left")
    tt = t[lo:hi]; ss = sat[lo:hi]
    bins = ((tt - start) // 60) * ns + ss
    out = []
    for c in cols:
        out.append(np.bincount(bins, weights=None if c is None else c[lo:hi],
                               minlength=60 * ns).reshape(60, ns).cumsum(axis=0))
    return out


def efim_constant(t, sat, g, starts, ns):
    """Position EFIM [m^-2-scaled by sigma] for every start and wait 1..60 min.
    Returns F (len(starts),60,3,3) WITHOUT the 1/sigma^2 factor."""
    ij = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    cols = [None] + [g[:, k] for k in range(3)] + [g[:, a] * g[:, b] for a, b in ij]
    F = np.empty((len(starts), 60, 3, 3))
    for u, s in enumerate(starts):
        m = _window_moments(t, sat, cols, s, ns)
        n = m[0]; sg = np.stack(m[1:4], axis=2)
        raw = np.empty((60, 3, 3))
        for z, (a, b) in enumerate(ij):
            raw[:, a, b] = m[4 + z].sum(axis=1); raw[:, b, a] = raw[:, a, b]
        inv = np.zeros_like(n, dtype=float); np.divide(1.0, n, out=inv, where=n > 0)
        F[u] = raw - np.einsum("wsi,wsj,ws->wij", sg, sg, inv, optimize=True)
    return (F + np.swapaxes(F, 2, 3)) / 2


def efim_general(t, sat, X, npos, starts, ns, waits, prior_prec=None):
    """General per-satellite Schur complement.

    X: (n_samples, p) design rows per sample, first `npos` columns belong to the
    receiver position, the remaining columns are per-satellite nuisance
    parameters (each satellite has its own copy).  prior_prec: length p-npos
    vector of prior precisions (0 = flat) for the nuisance parameters, already
    multiplied by sigma^2 so it can be added to unscaled moments.
    Returns F (len(starts), len(waits), npos, npos) without 1/sigma^2.
    """
    p = X.shape[1]; q = p - npos
    iu = [(a, b) for a in range(p) for b in range(a, p)]
    cols = [X[:, a] * X[:, b] for a, b in iu]
    pp = np.zeros(q) if prior_prec is None else np.asarray(prior_prec, float)
    widx = np.asarray(waits) - 1
    F = np.empty((len(starts), len(waits), npos, npos))
    for u, s in enumerate(starts):
        m = _window_moments(t, sat, cols, s, ns)
        M = np.empty((len(widx), ns, p, p))
        for z, (a, b) in enumerate(iu):
            M[:, :, a, b] = m[z][widx]; M[:, :, b, a] = M[:, :, a, b]
        A = M[:, :, :npos, :npos]; B = M[:, :, :npos, npos:]; C = M[:, :, npos:, npos:].copy()
        C += np.diag(pp)
        # pseudo-inverse handles satellites with too few samples (zero blocks)
        Ci = np.linalg.pinv(C, rcond=1e-12, hermitian=True)
        S = A - np.einsum("wsik,wskl,wsjl->wsij", B, Ci, B, optimize=True)
        F[u] = S.sum(axis=1)
    return (F + np.swapaxes(F, 2, 3)) / 2


def crlb_from_F(F, sigma=SIGMA):
    """Vectorised H/V/3D CRLB [m] from unscaled EFIM (... ,3,3) and the
    declared numerical-rank rule.  Non-regular windows -> inf."""
    J = F / sigma ** 2
    ew, ev = np.linalg.eigh(J)
    emax = np.maximum(ew[..., 2], 0)
    good = (ew[..., 0] > 1e-18) & (ew[..., 0] > emax * 1e-12)
    safe = np.where(good[..., None], ew, 1.0)
    P = np.einsum("...ik,...k,...jk->...ij", ev, 1.0 / safe, ev)
    Q = np.einsum("ab,...bc,dc->...ad", ROT, P, ROT)
    h = np.sqrt(np.maximum(Q[..., 0, 0] + Q[..., 1, 1], 0))
    v = np.sqrt(np.maximum(Q[..., 2, 2], 0))
    d = np.sqrt(np.maximum(Q[..., 0, 0] + Q[..., 1, 1] + Q[..., 2, 2], 0))
    for arr in (h, v, d):
        arr[~good] = np.inf
    return h, v, d, ew[..., 0], np.where(good, ew[..., 2] / np.where(good, ew[..., 0], 1), np.inf)


def efim_drift(t, sat, g, starts, ns, waits):
    """Constant bias + unknown linear drift per satellite (flat priors), exact
    closed form with window-relative time in hours:
      J_i = Sgg - Sg Sg^T/n - c c^T / Stt_c,
      c = Sgt - Sg St/n,  Stt_c = Stt - St^2/n.
    Satellites with fewer than 3 samples contribute nothing."""
    widx = np.asarray(waits) - 1
    F = np.empty((len(starts), len(waits), 3, 3))
    for u, s in enumerate(starts):
        lo = np.searchsorted(t, s, "left"); hi = np.searchsorted(t, s + 3600, "left")
        tau = (t[lo:hi] - s) / 3600.0
        gg = g[lo:hi]
        cols = [None, tau, tau * tau] + [gg[:, k] for k in range(3)] + [gg[:, k] * tau for k in range(3)] \
            + [gg[:, a] * gg[:, b] for a in range(3) for b in range(3)]
        tt = t[lo:hi]; ss = sat[lo:hi]
        bins = ((tt - s) // 60) * ns + ss
        m = [np.bincount(bins, weights=c, minlength=60 * ns).reshape(60, ns).cumsum(axis=0)[widx]
             for c in cols]
        n, St, Stt = m[0], m[1], m[2]
        Sg = np.stack(m[3:6], axis=2); Sgt = np.stack(m[6:9], axis=2)
        Sgg = np.stack(m[9:18], axis=2).reshape(len(widx), ns, 3, 3)
        ok = n >= 3
        nn = np.where(ok, n, 1.0)
        c = Sgt - Sg * (St / nn)[..., None]
        sttc = Stt - St ** 2 / nn
        ok &= sttc > 0
        sttc = np.where(ok, sttc, 1.0)
        Ji = Sgg - np.einsum("wsi,wsj->wsij", Sg, Sg) / nn[..., None, None] \
            - np.einsum("wsi,wsj->wsij", c, c) / sttc[..., None, None]
        Ji[~ok] = 0.0
        F[u] = Ji.sum(axis=1)
    return (F + np.swapaxes(F, 2, 3)) / 2
