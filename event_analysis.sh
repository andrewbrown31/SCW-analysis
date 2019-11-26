#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=6:00:00,mem=64GB
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.e

conda activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/event_analysis.py