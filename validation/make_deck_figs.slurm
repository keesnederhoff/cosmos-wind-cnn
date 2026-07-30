#!/usr/bin/env bash
# Focused figure run for the findings deck: per-station peak-event wind-field maps
# for the Alameda + USGS deep-dive slides. Writes to a _focus output dir so the
# published eraA/eraB results are untouched. Also builds the station skill maps.
#SBATCH --job-name=deck_figs
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=deck_figs_%j.log
set -euo pipefail

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

echo "=========== peak-event wind-field maps (era B focus) ==========="
# AAMC1 = Alameda (n=51k, the strongest Central Bay station); OAK is the nearest
# airport station; the three usable USGS moorings follow. EMC_MW101 is excluded --
# only ~28 h of CNN overlap, so its statistics are noise.
VAL_ERA=B VAL_USGS=1 VAL_SPATIAL=1 VAL_OUTDIR_SUFFIX=focus \
  VAL_STATIONS="AAMC1,OAK,ERO20_GRZ,WT_MW101,WT_MW201" \
  VAL_MODELS="ERA5,RTMA-SFbay,CNN-wave-p3-BC" \
  python run_validation.py

echo "=========== done ==========="
