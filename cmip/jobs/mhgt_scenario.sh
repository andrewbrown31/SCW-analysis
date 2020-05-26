#!/bin/bash

#PBS -P eg3 
#PBS -q hugemem
#PBS -l walltime=18:00:00,mem=512GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/mhgt_scenario.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/mhgt_scenario.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p mhgt -e rcp85 -m IPSL-CM5A-MR --force_compute True

