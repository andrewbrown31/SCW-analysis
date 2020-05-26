#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=2:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/srhe_left_scenario.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/srhe_left_scenario.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p srhe_left -e rcp85 -m ACCESS-ESM1-5 --force_compute True

