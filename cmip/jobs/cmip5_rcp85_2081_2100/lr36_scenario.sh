#!/bin/bash

#PBS -P eg3 
#PBS -q hugemem
#PBS -l walltime=24:00:00,mem=512GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/lr36_scenario_cmip5_rcp85_2081_2100.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/lr36_scenario_cmip5_rcp85_2081_2100.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p lr36 -e rcp85 --era5_y1 1979 --era5_y2 2005 --hist_y1 1979 --hist_y2 2005 --scenario_y1 2081 --scenario_y2 2100 --force_compute True

