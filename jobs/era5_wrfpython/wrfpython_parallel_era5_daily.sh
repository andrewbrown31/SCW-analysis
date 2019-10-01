#!/bin/bash

#PBS -P eg3 
#PBS -q normal 
#PBS -l walltime=00:30:00,mem=256GB 
#PBS -l ncpus=32
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_wrf_python.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_wrf_python.e 
#PBS -M andrew.brown@bom.gov.au 
#PBS -m ae 
 
 
#Set up conda/shell environments 
conda activate wrfpython3.6 
module load mpi4py/3.0.0-py3 
module unload python3/3.6.2 

#Initialise date
d=2016-09-01
#Specify end date
while [ "$d" != 2016-10-01 ]; do

  start_time=$(date -d "$d" +%Y%m%d)"00"
  end_time=$(date -d "$d"  +%Y%m%d)"23"
  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "era5" "aus" $start_time $end_time "True" "era5" 1

  #Advance date
  d=$(date -I -d "$d + 1 day")
done




