#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=6:00:00,mem=64GB
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/compare_mean.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/compare_mean.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/cmip/compare_mean.py
