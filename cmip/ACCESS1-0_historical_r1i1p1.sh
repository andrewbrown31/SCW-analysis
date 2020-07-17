#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=04:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_ACCESS1-0_historical.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_ACCESS1-0_historical.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38
 
#Set up conda/shell environments 
source activate wrfpython3.6 

#d=1960-01-01
#while [ "$d" != 2005-01-01 ]; do

#	start_time=$(date -d "$d" +%Y)"010100"
#	end_time=$(date -d "$d + 4 year"  +%Y)"123118"

python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m ACCESS1-0 -r aus -t1 2005010100 -t2 2005123118 --issave True --outname ACCESS1-0_historical_r1i1p1 -e historical --ens r1i1p1

#	d=$(date -I -d "$d + 5 year")

#done
