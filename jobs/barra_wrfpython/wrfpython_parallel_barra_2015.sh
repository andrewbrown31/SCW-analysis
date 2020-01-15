#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=48:00:00,mem=256GB 
#PBS -l ncpus=32
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_fc_wrf_python_2015.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_fc_wrf_python_2015.e 
#PBS -l other=gdata<x>
 
#Set up conda/shell environments 
conda activate wrfpython3.6 
module load mpi4py/3.0.0-py3 
module unload python3/3.6.2 

#Initialise date
d=2015-01-01
#Specify end date
while [ "$d" != 2016-01-01 ]; do

  start_time=$(date -d "$d" +%Y%m%d)"00"
  end_time=$(date -d "$d"  +%Y%m%d)"23"
  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
  mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" $start_time $end_time "True" "barra_fc" 1

  #REMOVE OLD MONTHLY FILES (done every day, but will only work the first day each month for one of the potential "month_end_date"s)
  month_start_date=$(date -d "$d" +%Y%m)"01"
  month_end_date1=$(date -d "$d" +%Y%m)"28"
  month_end_date2=$(date -d "$d" +%Y%m)"29"
  month_end_date3=$(date -d "$d" +%Y%m)"30"
  month_end_date4=$(date -d "$d" +%Y%m)"31"
  rm -f "/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc_${month_start_date}_${month_end_date1}.nc"
  rm -f "/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc_${month_start_date}_${month_end_date2}.nc"
  rm -f "/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc_${month_start_date}_${month_end_date3}.nc"
  rm -f "/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc_${month_start_date}_${month_end_date4}.nc"

  #Advance date
  d=$(date -I -d "$d + 1 day")

done


#Concaternate daily netcdf output into monthly files
module load cdo
path="/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/"
d=2015-01-01
while [ "$d" != 2016-01-01 ]; do

  files=$(date -d "$d" +%Y%m)
  start_date=$(date -d "$d" +%Y%m%d)
  temp_end_date=$(date -I -d "$d + 1 month - 1 day")
  end_date=$(date -d "$temp_end_date" +%Y%m%d)

  cdo -z zip_1 mergetime "${path}barra_fc_${files}"* "${path}monthly_${start_date}_${end_date}"
  rm "${path}barra_fc_${files}"*
  mv "${path}monthly_${start_date}_${end_date}" "${path}barra_fc_${start_date}_${end_date}.nc"

  #Advance date
  d=$(date -I -d "$d + 1 month")
done


