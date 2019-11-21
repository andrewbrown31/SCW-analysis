#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=06:00:00,mem=64GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_points_2011.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_points_2011.e 
 
#Set up conda/shell environments 
conda activate wrfpython3.6 

python /home/548/ab4502/working/ExtremeWind/barra_read.py 2011 2011 ml_cin

