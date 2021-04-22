#!/bin/bash

#PBS -P eg3
#PBS -q normal
#PBS -l walltime=24:00:00,mem=190GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/diurnal_compare_p.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/diurnal_compare_p.e
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate geopandas

python /home/548/ab4502/working/ExtremeWind/cmip/diurnal_compare_p.py
