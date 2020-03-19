#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=6:00:00,mem=64GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_GFDL-CM3.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_GFDL-CM3.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38+gdata/al33
 
#Set up conda/shell environments 
source activate wrfpython3.6 

d=1974-01-01
while [ "$d" != 2009-01-01 ]; do

	start_time=$(date -d "$d" +%Y)"010100"
	end_time=$(date -d "$d"  +%Y)"123118"

	python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m GFDL-CM3 -r aus -t1 $start_time -t2 $end_time --issave True --outname GFDL-CM3_historical_r1i1p1 -e historical --ens r1i1p1

	d=$(date -I -d "$d + 5 year")

done
