#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=4:00:00,mem=32GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/cmip_analysis.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/cmip_analysis.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38


source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_analysis.py -p dcp --plot False --lsm True

