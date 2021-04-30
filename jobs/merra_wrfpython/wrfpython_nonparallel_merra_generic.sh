#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=06:00:00,mem=64GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/merra_wrf_python_YEAR.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/merra_wrf_python_YEAR.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr7+gdata/ua8
 
#Set up conda/shell environments 
source activate wrfpython3.6

#Initialise date
d=YEAR-01-01
#Specify end date
while [ "$d" != YearPlusOne-01-01 ]; do

  start_time=$(date -d "$d" +%Y%m%d)"00"
  end_time=$(date -d "$d + 1 month - 1 day" +%Y%m%d)"18"
  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
  python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m merra2 -r aus -t1 $start_time -t2 $end_time --issave True --outname merra2 --params full --delta_t 3

  #Advance date
  d=$(date -I -d "$d + 1 month")

done


