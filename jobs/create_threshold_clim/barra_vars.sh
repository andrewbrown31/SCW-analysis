#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=32GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/event_analysis.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/event_analysis.py ml_cape 0 barra_fc 
python /home/548/ab4502/working/ExtremeWind/event_analysis.py ml_el 0 barra_fc 
python /home/548/ab4502/working/ExtremeWind/event_analysis.py s06 0 barra_fc 
python /home/548/ab4502/working/ExtremeWind/event_analysis.py Umean06 0 barra_fc 
