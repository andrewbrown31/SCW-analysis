#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=24:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario_cmip5_rcp85_2046_2080.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario_cmip5_rcp85_2046_2080.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m GFDL-CM3 --force_compute True --scenario_y1 2046 --scenario_y2 2080
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m GFDL-ESM2G --force_compute True --scenario_y1 2046 --scenario_y2 2080
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m GFDL-ESM2M --force_compute True --scenario_y1 2046 --scenario_y2 2080
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m IPSL-CM5A-LR --force_compute True --scenario_y1 2046 --scenario_y2 2080
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m IPSL-CM5A-MR --force_compute True --scenario_y1 2046 --scenario_y2 2080
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m MIROC5 --force_compute True --scenario_y1 2046 --scenario_y2 2080
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m MRI-CGCM3 --force_compute True --scenario_y1 2046 --scenario_y2 2080