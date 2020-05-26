#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=4:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_CNRM-CM5.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_CNRM-CM5.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38+gdata/al33
 
#Set up conda/shell environments 
source activate wrfpython3.6 

#d=1960-01-01
#while [ "$d" != 1970-01-01 ]; do

	#start_time=$(date -d "$d" +%Y)"010100"
	#end_time=$(date -d "$d"  +%Y)"123118"

python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m CNRM-CM5 -r aus -t1 1971010100 -t2 1971123118 --issave True --outname CNRM-CM5_historical_r1i1p1 -e historical --ens r1i1p1 --group CNRM-CERFACS --al33 True

#	d=$(date -I -d "$d + 1 year")

#done
