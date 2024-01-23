#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=12:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/erai_points.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/erai_points.e 
#PBS -lstorage=gdata/eg3+gdata/ma05+gdata/ub4
 
#Set up conda/shell environments 
source activate wrfpython3.6 

python /home/548/ab4502/working/ExtremeWind/erai_read.py 2005 2015

