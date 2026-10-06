#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares

C=299_792_458.0
MU=3.986004418e14
RE=6_371_000.0

@dataclass(frozen=True)
class Config:
    altitude_m: float=550_000.0
    cross_track_m: float=300_000.0
    carrier_hz: float=2.0e9
    sigma_rr_mps: float=0.2
    elevation_mask_deg: float=20.0
    dt_s: float=1.0
    total_time_s: float=30.0
    total_angle_deg: float=30.0
    min_gap_s: float=10.0
    bias_mps: float=25.0
    prior_sigma_m: float=20_000.0
    mc_trials: int=500
    random_trials: int=1000
    seed: int=20261006
    @property
    def v_mps(self): return math.sqrt(MU/(RE+self.altitude_m))

class Model:
    def __init__(self,cfg:Config):
        self.c=cfg
        self.v=cfg.v_mps
        g=np.linspace(0,600,20001)
        ok=np.array([self.el(t)>=cfg.elevation_mask_deg for t in g])
        self.tlim=float(g[np.where(ok)[0][-1]])
    def state(self,t):
        return np.array([self.v*t,self.c.cross_track_m,self.c.altitude_m]),np.array([self.v,0.,0.])
    def el(self,t):
        s,_=self.state(t); return math.degrees(math.atan2(s[2],math.hypot(s[0],s[1])))
    def azel(self,t):
        s,_=self.state(t)
        return math.degrees(math.atan2(s[0],s[1]))%360, self.el(t)
    def los(self,t):
        s,_=self.state(t); return s/np.linalg.norm(s)
    def sep(self,a,b):
        return math.degrees(math.acos(float(np.clip(self.los(a)@self.los(b),-1,1))))
    def rr(self,t,x=0.,y=0.,b=0.):
        s,vs=self.state(t); d=s-np.array([x,y,0.]); return float(d@vs/np.linalg.norm(d)+b)
    def doppler(self,t): return -self.rr(t)*self.c.carrier_hz/C
    def H(self,t):
        s,vs=self.state(t); d=s; rho=np.linalg.norm(d); u=d/rho
        grad=-((np.eye(3)-np.outer(u,u))@vs)/rho
        return np.array([grad[0],grad[1],1.0])

def crlb(model,times):
    H=np.vstack([model.H(float(t)) for t in times])
    J=H.T@H/(model.c.sigma_rr_mps**2)
    if np.linalg.eigvalsh(J)[0]<=1e-18: return (np.inf,np.inf,np.inf,np.inf)
    P=np.linalg.inv(J)
    return math.sqrt(P[0,0]+P[1,1]),math.sqrt(P[0,0]),math.sqrt(P[1,1]),np.linalg.cond(J)

def time_samples(model,centers,duration):
    out=[]
    for c in centers:
        t=np.arange(math.ceil(c-duration/2),math.floor(c+duration/2)+1,model.c.dt_s)
        out.extend(t[(t>=-model.tlim)&(t<=model.tlim)].tolist())
    return np.array(sorted(set(out)),float)

def centers_from_pos(pos,n):
    p=np.sort(np.asarray(pos,float))
    return np.r_[-p[::-1],p] if n%2==0 else np.r_[-p[::-1],[0.],p]

def optimize_time(model,n):
    dur=model.c.total_time_s/n
    if n==1:
        best=(np.inf,None)
        for c in np.arange(-model.tlim+dur/2,model.tlim-dur/2+0.1,1.0):
            v=crlb(model,time_samples(model,[c],dur))[0]
            if v<best[0]: best=(v,float(c))
        ctr=np.array([best[1]])
    else:
        m=n//2
        pos=np.linspace(max(dur,25.),min(model.tlim-dur,140.),m)
        cand=np.arange(dur/2,model.tlim-dur/2+0.1,1.0)
        def val(p):
            ctr=centers_from_pos(p,n); s=np.sort(ctr)
            if np.any(np.diff(s)-dur<model.c.min_gap_s-1e-9): return np.inf
            return crlb(model,time_samples(model,ctr,dur))[0]
        if not np.isfinite(val(pos)): pos=np.linspace(35.,min(model.tlim-dur,150.),m)
        best=val(pos)
        for _ in range(10):
            changed=False
            for j in range(m):
                bj=pos[j]
                for x in cand:
                    q=pos.copy(); q[j]=x; v=val(q)
                    if v<best-1e-9: best=v; bj=x
                if bj!=pos[j]: pos[j]=bj; changed=True
            if not changed: break
        ctr=centers_from_pos(pos,n)
    times=time_samples(model,ctr,dur)
    hs,ats,cts,cond=crlb(model,times)
    return ctr,times,dur,hs,ats,cts,cond

def angular_bounds(model,center,width):
    target=width/2
    f=lambda t:model.sep(t,center)-target
    lo=-model.tlim if f(-model.tlim)<=0 else brentq(f,-model.tlim,center-1e-6)
    hi=model.tlim if f(model.tlim)<=0 else brentq(f,center+1e-6,model.tlim)
    return lo,hi

def optimize_angle(model,n):
    width=model.c.total_angle_deg/n
    cand=[]
    for center in np.arange(-190,190.1,5.):
        try: lo,hi=angular_bounds(model,float(center),width)
        except ValueError: continue
        if lo<=-model.tlim+1e-3 or hi>=model.tlim-1e-3: continue
        t=np.arange(math.ceil(lo),math.floor(hi)+1,model.c.dt_s)
        if len(t)>=2: cand.append((float(center),lo,hi,t))
    def valid(ids):
        iv=sorted((cand[i][1],cand[i][2]) for i in ids)
        return all(iv[k+1][0]-iv[k][1]>=5. for k in range(len(iv)-1))
    def ev(ids):
        x=[]
        for i in ids: x.extend(cand[i][3].tolist())
        t=np.array(sorted(set(x)),float)
        return crlb(model,t)[0],t
    target=np.linspace(cand[0][0],cand[-1][0],n+2)[1:-1]
    ids=[]
    for x in target:
        for i in np.argsort([abs(z[0]-x) for z in cand]):
            i=int(i)
            if i not in ids and valid(ids+[i]): ids.append(i); break
    best,_=ev(ids)
    for _ in range(8):
        changed=False
        for p in range(n):
            cur=ids[p]; bi=cur
            for i in range(len(cand)):
                if i in ids and i!=cur: continue
                q=ids.copy(); q[p]=i
                if not valid(q): continue
                v,_=ev(q)
                if v<best-1e-8: best=v; bi=i
            if bi!=cur: ids[p]=bi; changed=True
        if not changed: break
    ids=sorted(ids,key=lambda i:cand[i][0]); hs,t=ev(ids)
    ats,cts,cond=crlb(model,t)[1:]
    return np.array([cand[i][0] for i in ids]),t,width,hs,ats,cts,cond

def random_time(model,n,trials):
    rng=np.random.default_rng(model.c.seed+1000+n); dur=model.c.total_time_s/n
    lo=-model.tlim+dur/2; hi=model.tlim-dur/2; vals=[]
    for _ in range(300000):
        if len(vals)>=trials: break
        ctr=np.sort(rng.uniform(lo,hi,n))
        if n>1 and np.any(np.diff(ctr)<dur+model.c.min_gap_s): continue
        v=crlb(model,time_samples(model,ctr,dur))[0]
        if np.isfinite(v): vals.append(v)
    return np.asarray(vals)

def mc(model,times,trials):
    rng=np.random.default_rng(model.c.seed+int(round(float(np.sum(times)))))
    truth=np.array([0.,0.,model.c.bias_mps])
    clean=np.array([model.rr(t,*truth) for t in times]); errs=[]
    for _ in range(trials):
        z=clean+rng.normal(0,model.c.sigma_rr_mps,len(times))
        x0=np.array([rng.normal(0,model.c.prior_sigma_m),rng.normal(0,model.c.prior_sigma_m),model.c.bias_mps+rng.normal(0,10)])
        def r(th): return (np.array([model.rr(t,*th) for t in times])-z)/model.c.sigma_rr_mps
        sol=least_squares(r,x0,bounds=([-150e3,-150e3,-200],[150e3,150e3,200]),max_nfev=500)
        if sol.success: errs.append(np.linalg.norm(sol.x[:2]))
    e=np.asarray(errs)
    return np.median(e),math.sqrt(np.mean(e**2)),np.quantile(e,.95)

def run(out:Path,cfg:Config):
    out.mkdir(parents=True,exist_ok=True); (out/"results").mkdir(exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    m=Model(cfg); ns=[1,2,3,4,5,6,8,10]; rows=[]; designs={}
    for n in ns:
        ctr,t,dur,hs,ats,cts,cond=optimize_time(m,n); designs[n]=(ctr,t,dur)
        rv=random_time(m,n,cfg.random_trials)
        row=dict(n_openings=n,duration_each_s=dur,total_samples=len(t),optimal_centers_s=";".join(f"{x:.2f}" for x in ctr),
                 horizontal_std_m=hs,along_track_std_m=ats,cross_track_std_m=cts,condition_number=cond,
                 random_median_horizontal_std_m=float(np.median(rv)),random_p10_horizontal_std_m=float(np.quantile(rv,.1)),
                 random_p90_horizontal_std_m=float(np.quantile(rv,.9)))
        if n in (1,2,3,4,8):
            med,rms,p95=mc(m,t,cfg.mc_trials); row.update(mc_median_error_m=med,mc_rms_error_m=rms,mc_p95_error_m=p95)
        rows.append(row)
    ft=pd.DataFrame(rows); ft.to_csv(out/"results/fixed_total_time.csv",index=False)
    gr=[]
    for n in (3,4):
        ctr,t,dur=designs[n]
        for i,c0 in enumerate(ctr,1):
            az,el=m.azel(c0)
            gr.append(dict(n_openings=n,aperture_index=i,center_time_s=c0,center_azimuth_deg=az,center_elevation_deg=el,
                           effective_track_width_deg=m.sep(c0-dur/2,c0+dur/2)))
    pd.DataFrame(gr).to_csv(out/"results/optimal_aperture_geometry.csv",index=False)
    ar=[]
    for n in ns:
        ctr,t,w,hs,ats,cts,cond=optimize_angle(m,n)
        ar.append(dict(n_openings=n,width_each_deg=w,total_samples=len(t),optimal_centers_s=";".join(f"{x:.1f}" for x in ctr),
                       horizontal_std_m=hs,along_track_std_m=ats,cross_track_std_m=cts,condition_number=cond))
    fa=pd.DataFrame(ar); fa.to_csv(out/"results/fixed_total_angular_budget.csv",index=False)
    tt=np.linspace(-m.tlim,m.tlim,1200); d3=designs[3]
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.plot(tt,[m.doppler(t)/1000 for t in tt])
    for c0 in d3[0]: ax.axvspan(c0-d3[2]/2,c0+d3[2]/2,alpha=.18)
    ax.set(xlabel="Time from closest approach (s)",ylabel="Geometric Doppler at 2 GHz (kHz)",title="Optimized three-aperture observation windows"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/"figures/fig1_doppler_three_apertures.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.plot(ft.n_openings,ft.horizontal_std_m,marker="o"); ax.set_yscale("log"); ax.grid(alpha=.25)
    ax.set(xlabel="Number of separated apertures",ylabel="CRLB horizontal 1-sigma (m)",title=f"Fixed total observation time = {cfg.total_time_s:.0f} s")
    fig.tight_layout(); fig.savefig(out/"figures/fig2_fixed_time_crlb.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.plot(ft.n_openings,ft.horizontal_std_m,marker="o",label="FIM-optimized"); ax.plot(ft.n_openings,ft.random_median_horizontal_std_m,marker="s",label="Random median")
    ax.fill_between(ft.n_openings,ft.random_p10_horizontal_std_m,ft.random_p90_horizontal_std_m,alpha=.15); ax.set_yscale("log"); ax.grid(alpha=.25); ax.legend()
    ax.set(xlabel="Number of separated apertures",ylabel="CRLB horizontal 1-sigma (m)",title="Aperture placement matters at equal observation time")
    fig.tight_layout(); fig.savefig(out/"figures/fig3_optimal_vs_random.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.plot(fa.n_openings,fa.horizontal_std_m,marker="o"); ax.set_yscale("log"); ax.grid(alpha=.25)
    ax.set(xlabel="Number of separated apertures",ylabel="CRLB horizontal 1-sigma (m)",title=f"Fixed total angular sky-track budget = {cfg.total_angle_deg:.0f} deg")
    fig.tight_layout(); fig.savefig(out/"figures/fig4_fixed_angle_crlb.png",dpi=200); plt.close(fig)
    print(ft[["n_openings","horizontal_std_m","random_median_horizontal_std_m"]].to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=Path("output_aperture_leo")); ap.add_argument("--mc-trials",type=int,default=500); ap.add_argument("--random-layout-trials",type=int,default=1000)
    a=ap.parse_args(); run(a.output_dir,Config(mc_trials=a.mc_trials,random_trials=a.random_layout_trials))
