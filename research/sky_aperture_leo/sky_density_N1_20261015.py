import sys, numpy as np, pandas as pd, math, time
sys.path.insert(0,'.')
import revision_common as rc
c=rc.load_cache(); day=(c['t']<rc.DAY)&(c['sky'][:,2]>math.sin(math.radians(5)))
sky=c['sky'][day].astype(np.float32); sat=c['sat'][day]; t=c['t'][day]
cos5=math.cos(math.radians(5))
rows=[]
t0=time.time()
for el in np.arange(10,90.1,2.5):
    azs=np.arange(0,360,5.0) if el<90 else [0.0]
    U=np.stack([rc.sky_unit(a,el) for a in azs]).astype(np.float32)
    hit=(sky@U.T)>=cos5
    for j,a in enumerate(azs):
        h=hit[:,j]
        rows.append((float(a),float(el),int(h.sum()),int(np.unique(sat[h]).size)))
df=pd.DataFrame(rows,columns=['az','el','n_obs','n_sat'])
df.to_csv(rc.ROOT/'results/revision_20261012/E9_sky_density_N1.csv',index=False)
print(time.time()-t0)
print(df.sort_values('n_obs',ascending=False).head(8).to_string(index=False))
print(df.sort_values('n_sat',ascending=False).head(5).to_string(index=False))
print(df[(df.az==0)&(df.el.isin([20,22.5,25]))].to_string(index=False))
