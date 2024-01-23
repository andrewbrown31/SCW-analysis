#!/bin/bash

for i in $(seq 1979 1 2020); do
 
 let j=i+1

 cp /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_01.sh
 cp /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_07.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_01.sh
 sed -i "s/YearPlusOne/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_01.sh
 sed -i "s/MMstart/01/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_01.sh
 sed -i "s/MMend/07/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_01.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_07.sh
 sed -i "s/YearPlusOne/$j/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_07.sh
 sed -i "s/MMstart/07/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_07.sh
 sed -i "s/MMend/01/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_07.sh

 qsub /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_01.sh
 qsub /home/548/ab4502/working/ExtremeWind/jobs/era5_global_wrfpython/wrfpython_parallel_era5_${i}_07.sh

 done


