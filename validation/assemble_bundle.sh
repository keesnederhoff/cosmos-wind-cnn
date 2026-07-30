#!/usr/bin/env bash
# Assemble the SF Bay validation bundle on Caldera via symlinks (no copies).
set -euo pipefail
B=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
V=$B/validation
OBS=$B/sf_bay_observed_data
RES=$B/sf_bay_rtma/results
RAW=$B/sf_bay_rtma/raw_data
C404=$B/sf_bay_conus404/raw_data

mkdir -p "$V"/obs "$V"/moorings "$V"/reference "$V"/cnn "$V"/era5 "$V"/conus404 "$V"/rtma "$V"/results

# obs + reference (explicit destination filenames -- some coreutils reject `ln -sf src dir/`)
ln -sf "$OBS"/pws_sfbay_waterfront_iem.nc        "$V"/obs/pws_sfbay_waterfront_iem.nc
ln -sf "$OBS"/pws_sfbay_waterfront_ndbc.nc       "$V"/obs/pws_sfbay_waterfront_ndbc.nc
ln -sf "$OBS"/pws_sfbay_waterfront_cwop_madis.nc "$V"/obs/pws_sfbay_waterfront_cwop_madis.nc
ln -sf "$OBS"/ERO20_GrizzlyBay_meteorological.nc "$V"/obs/ERO20_GrizzlyBay_meteorological.nc
# USGS moorings (config.MOORINGS_DIR = DATA_ROOT/"moorings"). ERO20 is read from
# PWS_DIR and already linked under obs/ above; these three are Whales Tale / EMC.
ln -sf "$OBS"/DMP23MW101met.nc "$V"/moorings/DMP23MW101met.nc
ln -sf "$OBS"/DMP23MW201met.nc "$V"/moorings/DMP23MW201met.nc
ln -sf "$OBS"/EMC26MW101met.nc "$V"/moorings/EMC26MW101met.nc

ln -sf "$OBS"/station_inventory.csv "$V"/reference/station_inventory.csv
ln -sf "$OBS"/station_inventory.md  "$V"/reference/station_inventory.md

# CNN winners (rename to the config's expected bundle names)
ln -sf "$RES"/os_av_bc24_terr_res_s2/output_inference/full_record_ERA5_20110101_20260101.nc "$V"/cnn/cnn_allvars.nc
ln -sf "$RES"/os_wo_bc24_base_res_s2/output_inference/full_record_ERA5_20110101_20260101.nc "$V"/cnn/cnn_windonly.nc
ln -sf "$RES"/x10_wo_bc24_res_d1_s2/output_inference/full_record_ERA5_20110101_20260101.nc "$V"/cnn/cnn_extreme.nc
ln -sf "$RES"/wv_wo_bc24_res_p2_w10_s1/output_inference/full_record_ERA5_20110101_20260101.nc "$V"/cnn/cnn_wave_p2.nc
ln -sf "$RES"/wv_wo_bc24_res_p3_w10_s2/output_inference/full_record_ERA5_20110101_20260101.nc "$V"/cnn/cnn_wave_p3.nc
ln -sf "$RES"/os_wo_bc24_base_res_s2/output_inference/BC_ERA5_20110101_20260101.nc "$V"/cnn/cnn_windonly_bc.nc
ln -sf "$RES"/os_av_bc24_terr_res_s2/output_inference/BC_ERA5_20110101_20260101.nc "$V"/cnn/cnn_allvars_bc.nc
ln -sf "$RES"/x10_wo_bc24_res_d1_s2/output_inference/BC_ERA5_20110101_20260101.nc "$V"/cnn/cnn_extreme_bc.nc
ln -sf "$RES"/wv_wo_bc24_res_p2_w10_s1/output_inference/BC_ERA5_20110101_20260101.nc "$V"/cnn/cnn_wave_p2_bc.nc
ln -sf "$RES"/wv_wo_bc24_res_p3_w10_s2/output_inference/BC_ERA5_20110101_20260101.nc "$V"/cnn/cnn_wave_p3_bc.nc

# baselines
ln -sf "$RAW"/ERA5_eastward_wind_1940_2026_UTM.nc  "$V"/era5/ERA5_eastward_wind_1940_2026_UTM.nc
ln -sf "$RAW"/ERA5_northward_wind_1940_2026_UTM.nc "$V"/era5/ERA5_northward_wind_1940_2026_UTM.nc
ln -sf "$C404"/CONUS404_SFbay_4km_eastward_wind_1979_2021_UTM10.nc  "$V"/conus404/CONUS404_SFbay_4km_eastward_wind_1979_2021_UTM10.nc
ln -sf "$C404"/CONUS404_SFbay_4km_northward_wind_1979_2021_UTM10.nc "$V"/conus404/CONUS404_SFbay_4km_northward_wind_1979_2021_UTM10.nc
ln -sf "$RAW"/RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc  "$V"/rtma/RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc
ln -sf "$RAW"/RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc "$V"/rtma/RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc

echo "=== bundle assembled at $V ==="
for d in obs moorings reference cnn era5 conus404 rtma; do echo "-- $d --"; ls -l "$V"/$d; done
