#!/usr/bin/env python3
"""Independent direct-centering spot checks against saved EFIM window outputs."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=HERE/"results/reproduced_3d_20261006"
LAT=math.radians(35.6662430056);LON=math.radians(139.7923087)
ROT=np.array([[-math.sin(LON),math.cos(LON),0],
  [-math.sin(LAT)*math.cos(LON),-math.sin(LAT)*math.sin(LON),math.cos(LAT)],
  [math.cos(LAT)*math.cos(LON),math.cos(LAT)*math.sin(LON),math.sin(LAT)]])
SIGMA=.2
rows=[]
for config in ["D1","D2","D3","D4","D5","D6","C6"]:
    obs=pd.read_csv(OUT/f"observations_{config}.csv.gz")
    cases=pd.read_csv(OUT/f"cases_{config}.csv.gz")
    # Use different starts and times; include relatively well-conditioned windows.
    for start,wait in [(0,60),(12000,60),(36000,60),(60000,60),(12000,30)]:
        subset=obs[(obs.t_offset_s>=start)&(obs.t_offset_s<start+wait*60)]
        J=np.zeros((3,3))
        for _,group in subset.groupby("norad_id"):
            g=group[["gx_sinv","gy_sinv","gz_sinv"]].to_numpy()
            gc=g-g.mean(axis=0)
            J+=gc.T@gc/(SIGMA**2)
        expected=cases[(cases.start_offset_s==start)&(cases.wait_min==wait)]
        assert len(expected)==1,(config,start,wait)
        exp=expected.iloc[0]
        saved=np.array([[exp.Jxx,exp.Jxy,exp.Jxz],[exp.Jxy,exp.Jyy,exp.Jyz],
                       [exp.Jxz,exp.Jyz,exp.Jzz]])
        rel=float(np.linalg.norm(J-saved)/(np.linalg.norm(J)+1e-40))
        eig=np.linalg.eigvalsh(J)
        p=np.linalg.eigvalsh(saved)
        finite=(eig[0]>1e-18)&(eig[0]>eig[-1]*1e-12)
        hdirect=math.inf
        if finite:
            Cov=np.linalg.inv(J)
            CovEnu=ROT@Cov@ROT.T
            hdirect=math.sqrt(max(0,CovEnu[0,0]+CovEnu[1,1]))
        recorded=exp.horizontal_crlb_m
        delta=abs(hdirect-recorded)/hdirect if np.isfinite(hdirect) and np.isfinite(recorded) else np.nan
        rows.append(dict(config=config,start_offset_s=start,wait_min=wait,nobs=len(subset),
            direct_rank_3=int(finite),relative_J_difference=rel,
            h_direct_m=hdirect,h_saved_m=recorded,relative_H_difference=delta))
        assert rel<5e-6,(config,start,wait,rel)
        if finite and np.isfinite(recorded) and eig[-1]/eig[0]<1e10:
            assert delta<.005,(config,start,wait,delta)
result=pd.DataFrame(rows)
result.to_csv(OUT/"direct_centering_validation.csv",index=False)
print(result.to_string(index=False))
print("DIRECT_CENTERING_VALIDATION_PASSED",len(rows),"windows")
