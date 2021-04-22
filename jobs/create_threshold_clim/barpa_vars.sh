#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=32GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_vars.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_vars.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

#python /home/548/ab4502/working/ExtremeWind/event_analysis.py scp_fixed 0 barpa_access
python /home/548/ab4502/working/ExtremeWind/event_analysis.py ml_cape 0 barpa_access
python /home/548/ab4502/working/ExtremeWind/event_analysis.py mu_cape 0 barpa_access
python /home/548/ab4502/working/ExtremeWind/event_analysis.py mlcape*s06 0 barpa_access
python /home/548/ab4502/working/ExtremeWind/event_analysis.py mucape*s06 0 barpa_access
#python /home/548/ab4502/working/ExtremeWind/event_analysis.py stp_fixed_left 0 barpa_access
