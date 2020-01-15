#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=24:00:00,mem=64GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_era5_sta.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_era5_sta.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/jobs/logit_selection/logit_era5_sta.py

