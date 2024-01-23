#!/bin/bash

#PBS -P eg3 
#PBS -q megamem
#PBS -l walltime=36:00:00,mem=2500GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/GFDL-CM3_global_laplacian_scenario_cmip5_rcp85_2081_2100.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/GFDL-CM3_global_laplacian_scenario_cmip5_rcp85_2081_2100.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38+gdata/rt52
#PBS -N GFDL-CM3_lap

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario_global_laplacian.py -e rcp85 -m GFDL-CM3 --era5_y1 1979 --era5_y2 2005 --hist_y1 1979 --hist_y2 2005 --scenario_y1 2081 --scenario_y2 2100 --force_compute True --save_hist_qm True --lsm False


