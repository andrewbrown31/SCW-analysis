#!/bin/bash

#PBS -P eg3 
#PBS -q hugemem
#PBS -l walltime=06:00:00,mem=512GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/t_totals_scenario.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/t_totals_scenario.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p t_totals -e rcp85 --threshold 48 --log False --vmin 0 --vmax 0.3

