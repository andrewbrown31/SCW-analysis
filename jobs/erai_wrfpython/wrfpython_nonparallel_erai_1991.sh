#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=12:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/erai_wrf_python_1991.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/erai_wrf_python_1991.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7
 
#Set up conda/shell environments 
source activate wrfpython3.6

#Initialise date
d=1991-01-01
#Specify end date
while [ "$d" != 1992-01-01 ]; do

  start_time=$(date -d "$d" +%Y%m%d)"00"
  end_time=$(date -d "$d + 1 month - 1 day" +%Y%m%d)"18"
  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
  python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m erai -r aus -t1 $start_time -t2 $end_time --issave True --outname erai --params min

  #Advance date
  d=$(date -I -d "$d + 1 month")

done


