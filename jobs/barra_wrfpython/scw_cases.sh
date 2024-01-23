#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=01:00:00,mem=64GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_fc_wrf_python_scw_cases.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_fc_wrf_python_scw_cases.e
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/cj37
#PBS -N 1990_01
 
#Set up conda/shell environments 
source activate wrfpython3.6 

#Initialise date
#declare -a dates=("20060924" "20091120" "20100306" "20100617" "20111118" "20111225" "20120226" "20130321" "20150228" "20100802" "20120810" "20130223" "20141014" "20150301" "20151216" "20160114" "20160604" "20170409" "20101216" "20110118" "20131123" "20161218" "20111007" "20131018" "20140123" "20160129" "20180213" "20071027" "20081209" "20101207" "20111108" "20120129" "20121130" "20141031" "20151207" "20171218")
declare -a dates=("20171219")

for d in "${dates[@]}"; do

  start_time=$(date -d "$d" +%Y%m%d)"00"
  end_time=$(date -d "$d"  +%Y%m%d)"23"
  file="/g/data/eg3/ab4502/ExtremeWind/vic/barra_fc/barra_fc_$(date -d "$d" +%Y%m%d)_$(date -d "$d"  +%Y%m%d).nc"
  if [ -e ${file} ]; then
     echo "INFO: FILE ALREADY EXISTS, GOING TO THE NEXT DAY,,,"
  else
     echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
     python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m barra_fc -r aus -t1 $start_time -t2 $end_time --issave True --outname barra_fc
  fi

done

