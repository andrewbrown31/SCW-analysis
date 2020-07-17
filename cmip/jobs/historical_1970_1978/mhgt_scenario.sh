#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=06:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/mhgt_scenario_historical_1970_1978.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/mhgt_scenario_historical_1970_1978.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

#python working/ExtremeWind/cmip/cmip_scenario.py -p mhgt -e historical --era5_y1 1979 --era5_y2 2005 --hist_y1 1979 --hist_y2 2005 --scenario_y1 1970 --scenario_y2 1978 --force_compute True -m ACCESS1-3 ACCESS1-0 BNU-ESM CNRM-CM5 GFDL-CM3 GFDL-ESM2G GFDL-ESM2M IPSL-CM5A-LR IPSL-CM5A-MR MIROC5 MRI-CGCM3 bcc-csm1-1 ACCESS-ESM1-5 ACCESS-CM2 --save_hist_qm True
python working/ExtremeWind/cmip/cmip_scenario.py -p mhgt -e historical --era5_y1 1979 --era5_y2 2005 --hist_y1 1979 --hist_y2 2005 --scenario_y1 1970 --scenario_y2 1978 --force_compute True -m ACCESS1-3 ACCESS1-0 --save_hist_qm True

