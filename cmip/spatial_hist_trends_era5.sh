#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=24:00:00,mem=32GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/spatial_hist_trends_era5_merra2.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/spatial_hist_trends_era5_merra2.e
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

python /home/548/ab4502/working/ExtremeWind/cmip/spatial_hist_trends_era5_merra2.py
