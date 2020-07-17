#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=2:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/mhgt_scenario_cmip6_historical_2006_2018.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/mhgt_scenario_cmip6_historical_2006_2018.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p mhgt -e historical --era5_y1 1979 --era5_y2 2005 --hist_y1 1979 --hist_y2 2005 --scenario_y1 2006 --scenario_y2 2014 --force_compute True -m ACCESS-ESM1-5 ACCESS-CM2

