#!/bin/bash

#PBS -P eg3
#PBS -q normal
#PBS -l walltime=24:00:00,mem=128GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/event_analysis.py
