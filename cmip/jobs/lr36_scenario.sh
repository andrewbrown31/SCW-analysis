#!/bin/bash

#PBS -P eg3 
#PBS -q hugemem
#PBS -l walltime=24:00:00,mem=1024GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/lr36_scenario.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/lr36_scenario.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p lr36 -e historical --era5_y1 1979 --era5_y2 2005 --hist_y1 1979 --hist_y2 2005 --scenario_y1 1970 --scenario_y2 1978 --force_compute True

