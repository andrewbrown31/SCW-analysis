#!/bin/bash

for i in $(seq 1990 1 2018); do
 
 let j=i+1

 cp /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_01.sh
 cp /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_05.sh
 cp /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_09.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_01.sh
 sed -i "s/YearPlusOne/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_01.sh
 sed -i "s/MMstart/01/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_01.sh
 sed -i "s/MMend/05/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_01.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_05.sh
 sed -i "s/YearPlusOne/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_05.sh
 sed -i "s/MMstart/05/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_05.sh
 sed -i "s/MMend/09/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_05.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_09.sh
 sed -i "s/YearPlusOne/$j/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_09.sh
 sed -i "s/MMstart/09/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_09.sh
 sed -i "s/MMend/01/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_09.sh

 qsub /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_01.sh
 qsub /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_05.sh
 qsub /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_${i}_09.sh

 done


