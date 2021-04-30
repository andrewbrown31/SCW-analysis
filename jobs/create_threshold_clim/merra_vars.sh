#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=64GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/merra2_vars.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/merra2_vars.e
#PBS -lstorage=gdata/eg3

python /home/548/ab4502/working/ExtremeWind/event_analysis.py t_totals 48.1 merra2
python /home/548/ab4502/working/ExtremeWind/event_analysis.py eff_sherb 0.47 merra2
python /home/548/ab4502/working/ExtremeWind/event_analysis.py dcp 0.15 merra2
python /home/548/ab4502/working/ExtremeWind/event_analysis.py bdsd 0.83 merra2


