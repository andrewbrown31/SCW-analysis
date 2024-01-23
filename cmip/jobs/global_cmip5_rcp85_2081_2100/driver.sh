#!/bin/bash

declare -a models=("ACCESS1-0" "ACCESS1-3" "bcc-csm1-1" "BNU-ESM" "CNRM-CM5" "GFDL-CM3" "GFDL-ESM2G" "GFDL-ESM2M" "IPSL-CM5A-LR" "IPSL-CM5A-MR" "MIROC5" "MRI-CGCM3")
declare -a vars=("mu_cape" "s06" "ta850" "ta500" "dp850")
path="/home/548/ab4502/working/ExtremeWind/cmip/jobs/global_cmip5_rcp85_2081_2100"

for m in "${models[@]}"; do
   for v in "${vars[@]}"; do
      cp $path"/generic_scenario.sh" $path"/"$m"/"$v"_scenario.sh"
      sed -i "s/MODEL/"$m"/g" $path"/"$m"/"$v"_scenario.sh"
      sed -i "s/DIAGNOSTIC/"$v"/g" $path"/"$m"/"$v"_scenario.sh"
      qsub $path"/"$m"/"$v"_scenario.sh"
   done
done
