#!/bin/sh 
  
#PBS -P eg3  
#PBS -q express
#PBS -l walltime=24:00:00,mem=128GB  
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/compress_barra.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/compress_barra.e 
#PBS -l other=gdata<x>

#This is a script which uses the ARCCSS CMS netcdf compression tool to compress BARRA 1-hourly convective parameters.

#This only has to be done becuase the data was concatenated from daily to monthly data without use of the cdo compression option.

module use /g/data3/hh5/public/modules
module load conda

nccompress -r -d 1 -b 5000 -o -pa -np 16 /g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/
