#!/bin/bash

for i in $(seq 1990 1 2018); do
 
 let j=i+1

 cp /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_$i.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_$i.sh
 sed -i "s/YearPlusOne/$j/g" /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_$i.sh

 qsub /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_$i.sh

 done


