#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=64GB
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/event_analysis.py logit 0.65 era5 is_sta lr36,ml_cape,srhe_left,Umean06
