#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=48:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_wrf_python_YEAR.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_wrf_python_YEAR.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7
 
#Set up conda/shell environments 
source activate wrfpython3.6 

#Initialise date
d=YEAR-01-01
#Specify end date
while [ "$d" != YearPlusFour-01-01 ]; do

  start_time=$(date -d "$d" +%Y%m%d)"00"
  end_time=$(date -d "$d + 1 month - 1 day"  +%Y%m%d)"18"

  if [ -f /g/data/eg3/ab4502/ExtremeWind/aus/barpa/barpa_access_$(date -d "$d" +%Y%m%d)_$(date -d "$d + 1 month - 1 day" +%Y%m%d).nc ]; then
    
    echo "INFO: FILE FOUND FOR " $start_time "to" $end_time

  else

    echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
    python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m barpa -r aus -t1 $start_time -t2 $end_time -e cmip5 --ens r1i1p1 --issave True --outname barpa_access --params reduced --barpa_forcing_mdl ACCESS1-0

  fi

  #Advance date
  d=$(date -I -d "$d + 1 month")

done


