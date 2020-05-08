#!/bin/sh

#PBS -P eg3
#PBS -q express
#PBS -l walltime=24:00:00,mem=32GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/plot_clim.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/plot_clim.e
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/plot_clim.py
