#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=64GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/obs_read.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/obs_read.e

conda activate sharppy

python /home/548/ab4502/working/ExtremeWind/obs_read.py
