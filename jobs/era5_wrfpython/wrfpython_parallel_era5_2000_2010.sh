#!/bin/bash

#PBS -P eg3
#PBS -q normal
#PBS -l walltime=48:00:00,mem=256GB
#PBS -l ncpus=32
#PBS -l other=gdata<x>
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_wrf_python_2000_2010.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_wrf_python_2000_2010.e
#PBS -M andrew.brown@bom.gov.au
#PBS -m abe

#Set up conda/shell environments
conda activate wrfpython3.6
module load mpi4py/3.0.0-py3
module unload python3/3.6.2

#Initialise date
d=2000-01-01
#Specify end date
while [ "$d" != 2011-01-01 ]; do

  start1=$(date -d "$d" +%Y%m%d)"00"
  start2=$(date -d "$d + 12 day" +%Y%m%d)"00"
  start3=$(date -d "$d + 24 day" +%Y%m%d)"00"

  end1=$(date -d "$d + 11 day" +%Y%m%d)"23"
  end2=$(date -d "$d + 23 day" +%Y%m%d)"23"
  end3=$(date -d "$d + 1 month - 1 day" +%Y%m%d)"23"

  month_end_date=$(date -I -d "$d + 1 month - 1 day")
  month_start_date=$(date -d "$d" +%Y%m%d)
  month_end_date=$(date -d "$month_end_date" +%Y%m%d)

  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start1 "to" $end1
 
  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "era5" "aus" $start1 $end1 "True" "era5" 1

  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start2 "to" $end2

  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "era5" "aus" $start2 $end2 "True" "era5" 1

  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start3 "to" $end3

  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "era5" "aus" $start3 $end3 "True" "era5" 1

  #Try to remove any pre-existing monthly files

  rm -f "/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5_${month_start_date}_${month_end_date}.nc"

  #Advance date
  d=$(date -I -d "$d + 1 month")
done

#Concaternate daily netcdf output into monthly files
module load cdo
path="/g/data/eg3/ab4502/ExtremeWind/aus/era5/"
d=2000-01-01
while [ "$d" != 2019-01-01 ]; do

  files=$(date -d "$d" +%Y%m)
  start_date=$(date -d "$d" +%Y%m%d)
  temp_end_date=$(date -I -d "$d + 1 month - 1 day")
  end_date=$(date -d "$temp_end_date" +%Y%m%d)

  cdo -z zip_1 mergetime "${path}era5_${files}"* "${path}monthly_${start_date}_${end_date}"
  rm "${path}era5_${files}"*
  mv "${path}monthly_${start_date}_${end_date}" "${path}era5_${start_date}_${end_date}.nc"

  #Advance date
  d=$(date -I -d "$d + 1 month")
done





