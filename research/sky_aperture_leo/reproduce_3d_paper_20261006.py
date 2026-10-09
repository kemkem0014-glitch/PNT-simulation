#!/usr/bin/env python3
"""Recompute manuscript Tables 8–12 from the FROZEN DTC OMM file.
No stochastic observations, no altitude constraint. Save case-level records.
Run: python research/sky_aperture_leo/reproduce_3d_paper_20261006.py
"""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec

ROOT = Path(__file__).resolve().parent
INFILE = ROOT / "results/tumsat_real_dtc_20261006/dtc_omm_snapshot.csv"
OUT = ROOT / "results/reproduced_3d_20261006"
OUT.mkdir(parents=True, exist_ok=True)
T0 = datetime(2026,10,6,tzinfo=timezone.utc).timestamp()
DAY = 86400
NSAT = 637
SIGMA = 0.2
LAT,LON,H_M = 35.6662430056,139.7923087,59.9
OMEGA = 7.2921150e-5
A = 6378.137
E2 = 6.69437999014e-3
RAD = np.pi/180
EOPT = math.degrees(math.atan(1/math.sqrt(2)))
WAITS = (5,10,15,20,30,45,60)
CW = (10,20,30,45,60)
STARTS = np.arange(0, DAY-3600+1,120,dtype=np.int64)
assert len(STARTS)==691

def observer():
    lat,lon=LAT*RAD,LON*RAD
    rn=A/math.sqrt(1-E2*math.sin(lat)**2)
    r=np.array([(rn+H_M/1000)*math.cos(lat)*math.cos(lon),
                (rn+H_M/1000)*math.cos(lat)*math.sin(lon),
                (rn*(1-E2)+H_M/1000)*math.sin(lat)])
    east=np.array([-math.sin(lon),math.cos(lon),0])
    north=np.array([-math.sin(lat)*math.cos(lon),-math.sin(lat)*math.sin(lon),math.cos(lat)])
    up=np.array([math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)])
    return r,np.stack((east,north,up))

R_OBS,ROT=observer()
def sky_unit(az,el):
    az,el=az*RAD,el*RAD
    return np.array([math.cos(el)*math.sin(az),math.cos(el)*math.cos(az),math.sin(el)])

CONF={}
for n in range(1,7):
    el=22.5 if n==1 else 45.0 if n==2 else EOPT
    centers=[(360*j/n,el) for j in range(n)]
    CONF[f"D{n}"]={"n":n,"kind":"distributed","centers":centers,"radius_deg":5.0}
    alpha=math.degrees(math.acos(1-n*(1-math.cos(5*RAD))))
    CONF[f"C{n}"]={"n":n,"kind":"concentrated","centers":[(0.0,22.5)],"radius_deg":alpha}
DIRECTIONS={k:np.stack([sky_unit(a,e) for a,e in c["centers"]]) for k,c in CONF.items()}

def julian(seconds):
    jd=np.asarray(seconds,dtype=float)/86400+2440587.5
    jd0=np.floor(jd)
    return jd0,jd-jd0,jd

def ecef(r,v,jd):
    d=(jd-2451545)/36525
    sec=67310.54841+(876600*3600+8640184.812866)*d+0.093104*d*d-6.2e-6*d*d*d
    th=np.deg2rad((sec/240)%360)
    co=np.cos(th);si=np.sin(th)
    x=co*r[:,0]+si*r[:,1]
    y=-si*r[:,0]+co*r[:,1]
    p=np.stack((x,y,r[:,2]),axis=1)
    xx=co*v[:,0]+si*v[:,1]+OMEGA*y
    yy=-si*v[:,0]+co*v[:,1]-OMEGA*x
    return p,np.stack((xx,yy,v[:,2]),axis=1)

def geom(r,v):
    d=r-R_OBS
    rho=np.linalg.norm(d,axis=1)
    los=d/rho[:,None]
    sky=los@ROT.T
    radial=np.einsum("ij,ij->i",los,v)
    grad=-(v-los*radial[:,None])/rho[:,None]
    return sky,grad,rho,radial

def segments(indices):
    if len(indices)==0:return []
    breaks=np.flatnonzero(np.diff(indices)>1)+1
    return np.split(indices,breaks)

def collect():
    with INFILE.open(encoding="utf-8") as f: fields=list(omm.parse_csv(f))
    fields=[f for f in fields if "[DTC]" in f.get("OBJECT_NAME","")]
    assert len(fields)==NSAT,(len(fields),NSAT)
    satrecs=[]
    for f in fields:
        obj=Satrec();omm.initialize(obj,f)
        satrecs.append(obj)
    offset=np.arange(0,DAY+1,60,dtype=np.int64)
    j0,jfr,j= julian(T0+offset)
    records={k:[] for k in CONF}
    skipped=[]
    for sidx,(f,sat) in enumerate(zip(fields,satrecs)):
        error,rr,vv=sat.sgp4_array(j0,jfr)
        ok=error==0
        if not np.any(ok):
            skipped.append(int(f["NORAD_CAT_ID"]));continue
        r,v=ecef(rr,vv,j)
        sky,_,_,_=geom(r,v)
        detectable=ok & (sky[:,2]>math.sin(5*RAD))
        if not np.any(detectable):continue
        spans=segments(np.flatnonzero(detectable))
        times=[]
        for sp in spans:
            lo=max(0,int(offset[sp[0]])-180)
            hi=min(DAY-1,int(offset[sp[-1]])+180)
            times.append(np.arange(lo,hi+1,dtype=np.int64))
        fine=np.unique(np.concatenate(times))
        jj0,jjf,jj=julian(T0+fine)
        err,pr,pv=sat.sgp4_array(jj0,jjf)
        valid=err==0
        if not np.any(valid):continue
        pp,vp=ecef(pr,pv,jj)
        sky,grad,rho,radial=geom(pp,vp)
        norad=int(f["NORAD_CAT_ID"])
        for name,cfg in CONF.items():
            U=DIRECTIONS[name]
            scores=sky@U.T
            cutoff=math.cos(cfg["radius_deg"]*RAD)
            hit=valid[:,None] & (scores>=cutoff)
            tt,ap=np.where(hit)
            if len(tt)==0:continue
            g=grad[tt];sk=sky[tt]
            entries=list(zip(
                fine[tt].astype(int).tolist(),
                [norad]*len(tt),
                (ap+1).tolist(),
                sk[:,0].tolist(),sk[:,1].tolist(),sk[:,2].tolist(),
                rho[tt].tolist(),radial[tt].tolist(),
                g[:,0].tolist(),g[:,1].tolist(),g[:,2].tolist()))
            records[name].extend(entries)
        if (sidx+1)%100==0:print("propagated",sidx+1,"/",len(fields),flush=True)
    columns=["t_offset_s","norad_id","aperture_index","los_e","los_n","los_u","range_km",
             "range_rate_km_s","gx_sinv","gy_sinv","gz_sinv"]
    data={}
    for name,entries in records.items():
        df=pd.DataFrame.from_records(entries,columns=columns)
        if len(df):
            df=df.drop_duplicates(["t_offset_s","norad_id","aperture_index"]).sort_values(
                ["t_offset_s","norad_id","aperture_index"]).reset_index(drop=True)
        data[name]=df
        df.to_csv(OUT/f"observations_{name}.csv.gz",index=False,compression="gzip",float_format="%.13g")
        print("observations",name,len(df),int(df.norad_id.nunique()) if len(df) else 0,flush=True)
    return data,skipped

def evaluate(df,waits,name):
    """Vectorized block moments per satellite, each window:
    EFIM=1/sigma^2 SUM_i (sum(gg^T) - sum(g)sum(g)^T/n_i).
    Rows are ECEF position gradients. No known-altitude constraint.
    """
    df=df.sort_values("t_offset_s")
    t=df.t_offset_s.to_numpy(dtype=np.int64)
    g=df[["gx_sinv","gy_sinv","gz_sinv"]].to_numpy(dtype=np.float64)
    satkeys=np.sort(df.norad_id.unique())
    sat=np.searchsorted(satkeys,df.norad_id.to_numpy())
    ns=len(satkeys)
    # 3 sums plus 6 quadratic moment terms for the 3x3 symmetric FIM
    ij=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    vals=[g[:,k] for k in range(3)]+[g[:,i]*g[:,j] for i,j in ij]
    rows=[]
    for u,start in enumerate(STARTS):
        lo=np.searchsorted(t,start,"left")
        hi=np.searchsorted(t,start+3600,"left")
        temp_t=t[lo:hi];temp_s=sat[lo:hi]
        # 60 minute bins: 1–60-minute windows share the same 60 min input subset.
        mins=(temp_t-start)//60
        bins=mins*ns+temp_s
        n=np.bincount(bins,minlength=60*ns).reshape(60,ns).cumsum(axis=0)
        moments=[np.bincount(bins,weights=z[lo:hi],minlength=60*ns)
                 .reshape(60,ns).cumsum(axis=0) for z in vals]
        sumg=np.stack(moments[:3],axis=2)
        sumgg=moments[3:]
        raw=np.empty((60,3,3))
        for z,(i,j) in enumerate(ij):
            raw[:,i,j]=sumgg[z].sum(axis=1)
            raw[:,j,i]=raw[:,i,j]
        inv_n=np.zeros_like(n,dtype=float)
        np.divide(1.0,n,out=inv_n,where=n>0)
        correction=np.einsum("wsi,wsj,ws->wij",sumg,sumg,inv_n,optimize=True)
        F=(raw-correction)/(SIGMA**2)
        F=(F+np.swapaxes(F,1,2))/2
        ew,ev=np.linalg.eigh(F)
        emax=np.maximum(ew[:,2],0)
        # Fixed, declared numerical-rank rule (NOT merely eig>0).
        good=(ew[:,0]>1e-18)&(ew[:,0]>emax*1e-12)
        ranks=np.sum(ew>np.maximum(1e-18,emax[:,None]*1e-12),axis=1)
        for w in waits:
            idx=w-1
            nobs=int(n[idx].sum())
            sn=int(np.count_nonzero(n[idx]))
            ns2=int(np.count_nonzero(n[idx]>=2))
            lam=np.array(ew[idx])
            if good[idx]:
                P=(ev[idx]/lam)@ev[idx].T
                Q=ROT@P@ROT.T
                h=math.sqrt(max(0,Q[0,0]+Q[1,1]))
                v=math.sqrt(max(0,Q[2,2]))
                d=math.sqrt(max(0,np.trace(Q)))
                cond=float(lam[-1]/lam[0])
            else:
                h=v=d=math.inf;cond=math.inf
            J=F[idx]
            rows.append((name,int(start),int(w),nobs,sn,ns2,int(ranks[idx]),
                float(lam[0]),float(lam[-1]),cond,h,v,d,
                float(J[0,0]),float(J[0,1]),float(J[0,2]),
                float(J[1,1]),float(J[1,2]),float(J[2,2])))
        if (u+1)%200==0:print("cases",name,u+1,"/",len(STARTS),flush=True)
    columns=["config","start_offset_s","wait_min","n_observations","n_satellites",
       "n_satellites_2plus","rank_ecef","eig_min","eig_max","condition_number",
       "horizontal_crlb_m","vertical_crlb_m","crlb_3d_m",
       "Jxx","Jxy","Jxz","Jyy","Jyz","Jzz"]
    return pd.DataFrame.from_records(rows,columns=columns)

def summarize_cases(df):
    rows=[]
    for (name,wait),g in df.groupby(["config","wait_min"]):
        finite=g[np.isfinite(g.horizontal_crlb_m)]
        rows.append(dict(config=name,wait_min=int(wait),n_cases=len(g),
             rank3_cases=len(finite),rank3_percent=100*len(finite)/len(g),
             median_horizontal_crlb_m=finite.horizontal_crlb_m.median(),
             p90_horizontal_crlb_m=finite.horizontal_crlb_m.quantile(.9),
             median_vertical_crlb_m=finite.vertical_crlb_m.median(),
             median_3d_crlb_m=finite.crlb_3d_m.median(),
             p90_3d_crlb_m=finite.crlb_3d_m.quantile(.9)))
    return pd.DataFrame(rows)

def main():
    data,skipped=collect()
    meta={"source_input":str(INFILE.relative_to(ROOT)),
          "source_sha256":hashlib.sha256(INFILE.read_bytes()).hexdigest(),
          "generated_utc":datetime.now(timezone.utc).isoformat(),
          "analysis_start_utc":"2026-10-06T00:00:00Z",
          "analysis_duration_hours":24,"n_dtc":NSAT,
          "skipped_norad":skipped,"receiver":{"lat_deg":LAT,"lon_deg":LON,"height_m":H_M},
          "coarse_step_s":60,"candidate_elevation_threshold_deg":5,
          "candidate_pad_s":180,"final_step_s":1,"sigma_range_rate_mps":SIGMA,
          "bias_model":"independent constant bias per satellite and evaluation window",
          "receiver_state":"ECEF x,y,z unknown, stationary receiver",
          "start_offset_s_step":120,"n_starts":len(STARTS),
          "waits_distributed_min":list(WAITS),"waits_concentrated_min":list(CW),
          "rank_rule":"eig_min>1e-18 and eig_min/eig_max>1e-12",
          "input_time_scale":"SGP4 UTC converted by JD approximation",
          "teme_to_ecef":"GMST with omega-earth velocity correction; no polar motion",
          "configs":CONF}
    (OUT/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    obsrows=[]
    for name,df in data.items():
        cfg=CONF[name]
        rec=dict(config=name,kind=cfg["kind"],n_openings=cfg["n"],
            diameter_deg=2*cfg["radius_deg"],
            n_observations=len(df),n_satellites=int(df.norad_id.nunique()) if len(df) else 0,
            per_aperture_satellites=";".join(str(v) for v in
                (df.groupby("aperture_index").norad_id.nunique().reindex(
                  range(1,cfg["n"]+1),fill_value=0).tolist())) if len(df) else "")
        obsrows.append(rec)
    counts=pd.DataFrame(obsrows)
    counts.to_csv(OUT/"table8_observation_counts.csv",index=False)
    parts=[]
    for name,df in data.items():
        # Old main 3-aperture analysis uses all 1-60 min; save independently.
        waits=range(1,61) if name=="D3" else CW if name.startswith("C") else WAITS
        dfcase=evaluate(df,waits,name)
        dfcase.to_csv(OUT/f"cases_{name}.csv.gz",index=False,compression="gzip",float_format="%.12g")
        if name=="D3":
            dfcase[dfcase.wait_min.isin(WAITS)].to_csv(
                OUT/"cases_D3_seven_waits.csv.gz",index=False,compression="gzip")
        parts.append(dfcase)
    cases=pd.concat(parts,ignore_index=True)
    seven=cases[((cases.config.str.startswith("D")) & (cases.wait_min.isin(WAITS)))]
    seven.to_csv(OUT/"all_distributed_29022_cases.csv.gz",
        index=False,compression="gzip",float_format="%.12g")
    summary=summarize_cases(cases)
    summary.to_csv(OUT/"all_wait_summaries.csv",index=False,float_format="%.12g")
    summary[summary.config.str.startswith("D") & summary.wait_min.isin(WAITS)].to_csv(
        OUT/"table9_rank3_percent.csv",index=False)
    summary[summary.config.str.startswith("D") & summary.wait_min.isin([10,30,60])].to_csv(
        OUT/"table10_crlb_medians.csv",index=False)
    attain=[]
    for name in [f"D{i}" for i in range(1,7)]:
        sub=summary[(summary.config==name)&(summary.wait_min.isin(WAITS))]
        row={"config":name}
        for limit in [100,50,20]:
            passing=sub[sub.median_horizontal_crlb_m<=limit]
            row[f"H_le_{limit}m_first_wait_min"]=int(passing.wait_min.min()) if len(passing) else ""
        attain.append(row)
    pd.DataFrame(attain).to_csv(OUT/"table11_threshold_waits.csv",index=False)
    comp=[]
    for n in range(1,7):
        A=counts.set_index("config")
        S=summary.set_index(["config","wait_min"])
        comp.append(dict(N=n,large_diameter_deg=float(A.loc[f"C{n}","diameter_deg"]),
            large_satellites=int(A.loc[f"C{n}","n_satellites"]),
            distributed_satellites=int(A.loc[f"D{n}","n_satellites"]),
            large_samples=int(A.loc[f"C{n}","n_observations"]),
            distributed_samples=int(A.loc[f"D{n}","n_observations"]),
            large_H60_m=S.loc[(f"C{n}",60),"median_horizontal_crlb_m"],
            distributed_H60_m=S.loc[(f"D{n}",60),"median_horizontal_crlb_m"],
            large_3D60_m=S.loc[(f"C{n}",60),"median_3d_crlb_m"],
            distributed_3D60_m=S.loc[(f"D{n}",60),"median_3d_crlb_m"]))
    pd.DataFrame(comp).to_csv(OUT/"table12_fixed_solid_angle.csv",index=False)
    print("OUTPUT",OUT, "all distributed cases",len(seven),flush=True)
    print(counts.to_string(index=False),flush=True)
    print(pd.DataFrame(comp).to_string(index=False),flush=True)

if __name__=="__main__":
    main()
