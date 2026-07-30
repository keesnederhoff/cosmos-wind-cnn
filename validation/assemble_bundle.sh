#!/usr/bin/env bash
# Assemble the SF Bay validation bundle on Caldera via symlinks (no copies).
# Layout mirrors config.py: observed_data/, reference/, modeled_data/{cnn_fullrecord,era5,conus404,rtma}/
set -euo pipefail
B=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
V=$B/validation
OBS=$B/sf_bay_observed_data
RES=$B/sf_bay_rtma/results
RAW=$B/sf_bay_rtma/raw_data
C404=$B/sf_bay_conus404/raw_data
M=$V/modeled_data

mkdir -p "$V"/observed_data "$V"/reference "$M"/cnn_fullrecord "$M"/era5 "$M"/conus404 "$M"/rtma "$V"/results

# observations (explicit destination filenames -- some coreutils reject `ln -sf src dir/`)
for f in pws_sfbay_waterfront_iem.nc pws_sfbay_waterfront_ndbc.nc \
         pws_sfbay_waterfront_cwop_madis.nc ERO20_GrizzlyBay_meteorological.nc \
         DMP23MW101met.nc DMP23MW201met.nc EMC26MW101met.nc; do
  ln -sf "$OBS/$f" "$V/observed_data/$f"
done
ln -sf "$OBS"/station_inventory.csv "$V"/reference/station_inventory.csv
ln -sf "$OBS"/station_inventory.md  "$V"/reference/station_inventory.md

# CNN products (bundle names the config expects)
link_cnn () { ln -sf "$RES/$1/output_inference/$2" "$M/cnn_fullrecord/$3"; }
link_cnn os_av_bc24_terr_res_s2   full_record_ERA5_20110101_20260101.nc cnn_allvars.nc
link_cnn os_wo_bc24_base_res_s2   full_record_ERA5_20110101_20260101.nc cnn_windonly.nc
link_cnn x10_wo_bc24_res_d1_s2    full_record_ERA5_20110101_20260101.nc cnn_extreme.nc
link_cnn wv_wo_bc24_res_p2_w10_s1 full_record_ERA5_20110101_20260101.nc cnn_wave_p2.nc
link_cnn wv_wo_bc24_res_p3_w10_s2 full_record_ERA5_20110101_20260101.nc cnn_wave_p3.nc
link_cnn os_av_bc24_terr_res_s2   BC_ERA5_20110101_20260101.nc cnn_allvars_bc.nc
link_cnn os_wo_bc24_base_res_s2   BC_ERA5_20110101_20260101.nc cnn_windonly_bc.nc
link_cnn x10_wo_bc24_res_d1_s2    BC_ERA5_20110101_20260101.nc cnn_extreme_bc.nc
link_cnn wv_wo_bc24_res_p2_w10_s1 BC_ERA5_20110101_20260101.nc cnn_wave_p2_bc.nc
link_cnn wv_wo_bc24_res_p3_w10_s2 BC_ERA5_20110101_20260101.nc cnn_wave_p3_bc.nc

# Upstream config points CNN-allvars / CNN-windonly at per-run dirs under
# modeled_data/<run>/ (and reads CNN-allvars' scalar channels from the same file),
# so mirror that layout as well as the cnn_fullrecord/ bundle names.
for r in os_av_bc24_terr_res_s2 os_wo_bc24_base_res_s2; do
  mkdir -p "$M/$r"
  ln -sf "$RES/$r/output_inference/full_record_ERA5_20110101_20260101.nc" \
         "$M/$r/full_record_ERA5_20110101_20260101.nc"
done

# baselines
# NOTE the air_temperature files: the engine opens each model's temp_file even in
# wind-only mode, and a missing one makes the whole product fail the path audit and
# vanish from the run silently. This has bitten twice -- keep them linked.
for f in ERA5_eastward_wind_1940_2026_UTM.nc ERA5_northward_wind_1940_2026_UTM.nc; do
  ln -sf "$RAW/$f" "$M/era5/$f"
done
ln -sf "$C404/ERA5_air_temperature_1940_2026_UTM.nc" \
       "$M/era5/ERA5_air_temperature_1940_2026_UTM.nc"
for f in CONUS404_SFbay_4km_eastward_wind_1979_2021_UTM10.nc \
         CONUS404_SFbay_4km_northward_wind_1979_2021_UTM10.nc \
         CONUS404_SFbay_4km_air_temperature_1979_2021_UTM10.nc; do
  ln -sf "$C404/$f" "$M/conus404/$f"
done
for f in RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc \
         RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc \
         RTMA_SFbay_2p5km_static_landsea_static_UTM10.nc; do
  ln -sf "$RAW/$f" "$M/rtma/$f"
done

echo "=== bundle assembled at $V ==="
for d in observed_data reference modeled_data/cnn_fullrecord modeled_data/era5 \
         modeled_data/conus404 modeled_data/rtma; do
  echo "-- $d --"; ls -l "$V/$d"
done
