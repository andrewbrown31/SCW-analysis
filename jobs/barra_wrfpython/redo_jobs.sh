#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=06:00:00,mem=256GB 
#PBS -l ncpus=32
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/redo_jobs.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/redo_jobs.e 

#Set up conda/shell environments 
conda activate wrfpython3.6
module load mpi4py/3.0.0-py3
module unload python3/3.6.2
module load cdo

d1=2002-07-01
d=2002-07-13
start_time_out=$(date -d "$d" +%Y%m%d)
end_time_out=$(date -d "$d"  +%Y%m%d)	
start_date=$(date -d "$d1"  +%Y%m%d)	
start_time=$(date -d "$d" +%Y%m%d)"00"
end_time=$(date -d "$d"  +%Y%m%d)"23"
temp_end_date=$(date -I -d "$d1 + 1 month - 1 day")
end_date=$(date -d "$temp_end_date" +%Y%m%d)
echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" $start_time $end_time "True" "barra_fc" 1
path="/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/"
cdo -z zip_1 mergetime "${path}barra_fc_${start_time_out}_${end_time_out}.nc" "${path}barra_fc_${start_date}_${end_date}.nc" "${path}temp_monthly"
mv "${path}temp_monthly" "${path}barra_fc_${start_date}_${end_date}.nc"
rm "${path}barra_fc_${start_time_out}_${end_time_out}.nc"
rm "${path}temp_monthly"

d1=2002-07-01
d=2002-07-14
start_time_out=$(date -d "$d" +%Y%m%d)
end_time_out=$(date -d "$d"  +%Y%m%d)	
start_date=$(date -d "$d1"  +%Y%m%d)	
start_time=$(date -d "$d" +%Y%m%d)"00"
end_time=$(date -d "$d"  +%Y%m%d)"23"
temp_end_date=$(date -I -d "$d1 + 1 month - 1 day")
end_date=$(date -d "$temp_end_date" +%Y%m%d)
echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" $start_time $end_time "True" "barra_fc" 1
path="/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/"
cdo -z zip_1 mergetime "${path}barra_fc_${start_time_out}_${end_time_out}.nc" "${path}barra_fc_${start_date}_${end_date}.nc" "${path}temp_monthly"
mv "${path}temp_monthly" "${path}barra_fc_${start_date}_${end_date}.nc"
rm "${path}barra_fc_${start_time_out}_${end_time_out}.nc"
rm "${path}temp_monthly"

d1=2018-07-01
d=2018-07-14
start_time_out=$(date -d "$d" +%Y%m%d)
end_time_out=$(date -d "$d"  +%Y%m%d)	
start_date=$(date -d "$d1"  +%Y%m%d)	
start_time=$(date -d "$d" +%Y%m%d)"00"
end_time=$(date -d "$d"  +%Y%m%d)"23"
temp_end_date=$(date -I -d "$d1 + 1 month - 1 day")
end_date=$(date -d "$temp_end_date" +%Y%m%d)
echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" $start_time $end_time "True" "barra_fc" 1
path="/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/"
cdo -z zip_1 mergetime "${path}barra_fc_${start_time_out}_${end_time_out}.nc" "${path}barra_fc_${start_date}_${end_date}.nc" "${path}temp_monthly"
mv "${path}temp_monthly" "${path}barra_fc_${start_date}_${end_date}.nc"
rm "${path}barra_fc_${start_time_out}_${end_time_out}.nc"
rm "${path}temp_monthly"

