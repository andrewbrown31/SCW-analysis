#!/bin/bash

for i in $(seq 1997 1 2004); do
 
 cp /home/548/ab4502/working/ExtremeWind/jobs/era5_points/era5_points_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/era5_points/era5_points_$i.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/era5_points/era5_points_$i.sh

 qsub /home/548/ab4502/working/ExtremeWind/jobs/era5_points/era5_points_$i.sh

 done


