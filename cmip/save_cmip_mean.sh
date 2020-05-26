#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=24:00:00,mem=128GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/save_cmip_mean.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/save_cmip_mean.e
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/cmip/save_cmip_mean.py
