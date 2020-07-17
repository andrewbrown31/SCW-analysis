#!/bin/bash

for i in $(seq 1990 5 2010); do
 
 let j=i+5

 cp /home/548/ab4502/working/ExtremeWind/jobs/barpa_wrfpython/wrfpython_barpa_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/barpa_wrfpython/wrfpython_barpa_$i.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/barpa_wrfpython/wrfpython_barpa_$i.sh
 sed -i "s/YearPlusFour/$j/g" /home/548/ab4502/working/ExtremeWind/jobs/barpa_wrfpython/wrfpython_barpa_$i.sh

 qsub /home/548/ab4502/working/ExtremeWind/jobs/barpa_wrfpython/wrfpython_barpa_$i.sh

 done


