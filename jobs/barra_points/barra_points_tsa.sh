#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=6:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_points_tsa.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_points_tsa.e 
#PBS -lstorage=gdata/eg3+gdata/ma05
 
#Set up conda/shell environments 
source activate wrfpython3.6 

python /home/548/ab4502/working/ExtremeWind/barra_read.py

