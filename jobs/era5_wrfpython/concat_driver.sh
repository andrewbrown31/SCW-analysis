#!/bin/bash

arr=(1979 1984 1989 1994 1999 2004 2009 2014)

for i in "${arr[@]}"; do
 
 let j=i+5

 cp /home/548/ab4502/working/ExtremeWind/jobs/era5_wrfpython/concat_era5_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/era5_wrfpython/concat_era5_$i.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_wrfpython/concat_era5_$i.sh
 sed -i "s/YearEnd/$j/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_wrfpython/concat_era5_$i.sh

 #qsub /home/548/ab4502/working/ExtremeWind/jobs/barra_wrfpython/wrfpython_parallel_barra_$i.sh


 done



