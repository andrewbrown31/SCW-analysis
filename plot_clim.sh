#!/bin/sh

#PBS -P eg3
#PBS -q express
#PBS -l walltime=02:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/plot_clim.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/plot_clim.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/plot_clim.py
