#!/bin/bash

declare -a models=("ACCESS1-0" "ACCESS1-3" "bcc-csm1-1" "BNU-ESM" "CNRM-CM5" "GFDL-CM3" "GFDL-ESM2G" "GFDL-ESM2M" "IPSL-CM5A-LR" "IPSL-CM5A-MR" "MIROC5" "MRI-CGCM3")
path="/home/548/ab4502/working/ExtremeWind/cmip/jobs/global_cmip5_rcp85_2081_2100"
v="t_totals"

for m in "${models[@]}"; do
      cp $path"/generic_scenario_t_totals.sh" $path"/"$m"/"$v"_scenario.sh"
      sed -i "s/MODEL/"$m"/g" $path"/"$m"/"$v"_scenario.sh"
      qsub $path"/"$m"/"$v"_scenario.sh"
done
