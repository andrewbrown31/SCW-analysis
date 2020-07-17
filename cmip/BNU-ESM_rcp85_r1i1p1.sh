#!/bin/bash

#PBS -P eg3 
#PBS -q hugemem
#PBS -l walltime=36:00:00,mem=1024GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_BNU-ESM_rcp85.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_BNU-ESM_rcp85.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38+gdata/al33
 
#Set up conda/shell environments 
source activate wrfpython3.6 

python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m BNU-ESM -r aus -t1 2026010100 -t2 2045123123 --issave True --outname BNU-ESM_rcp85_r1i1p1 -e rcp85 --ens r1i1p1
python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m BNU-ESM -r aus -t1 2081010100 -t2 2100123123 --issave True --outname BNU-ESM_rcp85_r1i1p1 -e rcp85 --ens r1i1p1

