#!/bin/bash
#SBATCH --job-name=sfb_frcat
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --account=pcmsc
#SBATCH --time=10:00:00
#SBATCH --output=sweep_logs/%x_%A.log
set -euo pipefail
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
python - <<'PY'
import os, numpy as np, xarray as xr
outdir="/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma/results/sw_bc24do0/output_inference"
segs=[]
for k in range(22):
    ys=1940+4*k; ye=min(ys+4,2027)
    f=f"{outdir}/full_record_ERA5_{ys}0101_{ye}0101.nc"
    if not os.path.exists(f):
        print("MISSING", f); continue
    d=xr.open_dataset(f, chunks={"time":2000})
    d=d.isel(time=slice(5,None))          # drop seq_len-1 NaN warmup pad
    segs.append(d)
print("segments:", len(segs))
ds=xr.concat(segs, dim="time")
t=ds.time.values
_, idx=np.unique(t, return_index=True)    # dedupe ~24h boundary overlap
ds=ds.isel(time=np.sort(idx))
dt=np.diff(ds.time.values).astype("timedelta64[s]").astype(int)
assert (dt>0).all(), "time not monotonic after dedupe"
print("merged:", ds.sizes["time"], "steps |", str(ds.time.values[0])[:19], "->", str(ds.time.values[-1])[:19])
enc={v:{"zlib":True,"complevel":1} for v in ds.data_vars if ds[v].ndim==3}
target=f"{outdir}/FULL_1940_2027_merged.nc"
ds.to_netcdf(target, encoding=enc)
print("WROTE", target, round(os.path.getsize(target)/1e9,1), "GB")
PY
