#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=128GB
#PBS -l ncpus=1
#PBS -l other=gdata<x>
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/concat_era5_1999.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/concat_era5_1999.e

#Concaternate daily netcdf output into monthly files
module load cdo
path="/g/data/eg3/ab4502/ExtremeWind/aus/era5/"
d=1999-01-01
while [ "$d" != 2000-01-01 ]; do

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




