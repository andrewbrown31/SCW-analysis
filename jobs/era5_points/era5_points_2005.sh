#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=04:00:00,mem=16GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_points_2005.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_points_2005.e 
 
#Set up conda/shell environments 
conda activate wrfpython3.6 

python /home/548/ab4502/working/ExtremeWind/era5_read.py 2005 2005

