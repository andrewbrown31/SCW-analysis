#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=00:30:00,mem=256GB
#PBS -l ncpus=32
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/erai_fc_wrf_python.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/erai_fc_wrf_python.e
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

  start1=$(date -d "$d" +%Y%m%d)"00"
  start2=$(date -d "$d + 12 day" +%Y%m%d)"00"
  start3=$(date -d "$d + 24 day" +%Y%m%d)"00"

  end1=$(date -d "$d + 11 day" +%Y%m%d)"23"
  end2=$(date -d "$d + 23 day" +%Y%m%d)"23"
  end3=$(date -d "$d + 1 month - 1 day" +%Y%m%d)"23"

  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start1 "to" $end1
 
  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "erai" "aus" $start1 $end1 "True" "wrf_python"

  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start2 "to" $end2

  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "erai" "aus" $start2 $end2 "True" "wrf_python"

  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start3 "to" $end3

  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "erai" "aus" $start3 $end3 "True" "wrf_python"

  #Advance date
  d=$(date -I -d "$d + 1 month")
done




