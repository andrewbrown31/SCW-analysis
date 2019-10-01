#!/bin/sh

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=06:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/concat_barra.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/concat_barra.e


#CONCATERNATE DAILY BARRA-R FORECAST CONVECTIVE PARAMETER OUTPUT TO MONTHLY,
# USING CDO


module load cdo

path="/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/"

#Initialise date
d=2014-09-01
#Specify end date
while [ "$d" != 2014-10-01 ]; do

  files=$(date -d "$d" +%Y%m)
  start_date=$(date -d "$d" +%Y%m%d)
  temp_end_date=$(date -I -d "$d + 1 month - 1 day")
  end_date=$(date -d "$temp_end_date" +%Y%m%d)
  
  cdo mergetime "${path}barra_fc_${files}"* "${path}monthly_${start_date}_${end_date}"
  rm "${path}barra_fc_${files}"*
  mv "${path}monthly_${start_date}_${end_date}" "${path}barra_fc_${start_date}_${end_date}.nc"

  #Advance date
  d=$(date -I -d "$d + 1 month")
done

