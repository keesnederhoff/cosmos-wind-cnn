#!/bin/bash
#SBATCH --job-name=cut2025
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=pcmsc
#SBATCH --time=1:00:00
#SBATCH --output=sweep_logs/%x_%A.log
set -euo pipefail
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
python - <<'PY'
import xarray as xr, numpy as np, os
RUN="sw_bc8do0"; YEAR=2025
base=f"/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma/results/{RUN}/output_inference"
src=f"{base}/full_record_ERA5_20110101_20260101.nc"
dst=f"{base}/{RUN}_{YEAR}_subset.nc"
ds=xr.open_dataset(src, chunks={"time":720})
sub=ds.sel(time=slice(f"{YEAR}-01-01", f"{YEAR}-12-31T23:00"))
print("timesteps:", sub.sizes["time"], "|", str(sub.time.values[0])[:19], "->", str(sub.time.values[-1])[:19])
assert sub.sizes["time"] > 8000
enc={v:{"zlib":True,"complevel":4} for v in sub.data_vars if sub[v].ndim==3}
sub.attrs=dict(ds.attrs); sub.attrs["subset_of"]=os.path.basename(src); sub.attrs["subset_year"]=str(YEAR)
sub.to_netcdf(dst, encoding=enc)
print("WROTE", dst, round(os.path.getsize(dst)/1e9,2), "GB")
chk=xr.open_dataset(dst); sp=np.sqrt(chk.hr_u**2+chk.hr_v**2)
print(f"  vars: {[v for v in chk.data_vars]}")
print(f"  wind speed mean={float(sp.mean()):.2f} max={float(sp.max()):.2f} m/s | finite={float(np.isfinite(sp).mean())*100:.1f}%")
PY

cd /caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma/results/sw_bc8do0/output_inference
echo "--- zipping ---"
zip -1 -j sw_bc8do0_2025_subset.zip sw_bc8do0_2025_subset.nc
ls -la sw_bc8do0_2025_subset.nc sw_bc8do0_2025_subset.zip | awk '{printf "  %-40s %6.2f GB\n", $9, $5/1e9}'
unzip -t sw_bc8do0_2025_subset.zip | tail -2
