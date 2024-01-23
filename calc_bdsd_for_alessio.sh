#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=04:00:00,mem=32GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/calc_bdsd_for_alessio.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/calc_bdsd_for_alessio.e
#PBS -l storage=gdata/eg3+gdata/hh5

module use /g/data/hh5/public/modules
module load conda/analysis3

python /home/548/ab4502/working/ExtremeWind/calc_bdsd_for_alessio.py

