#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=64GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_vars.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_vars.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/event_analysis.py t_totals 48.1 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py eff_sherb 0.47 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py logit 0.83 era5 is_conv_aws ebwd,Umean800_600,lr13,rhmin13,srhe_left,q_melting,eff_lcl
python /home/548/ab4502/working/ExtremeWind/event_analysis.py dcp 0.15 era5


python /home/548/ab4502/working/ExtremeWind/event_analysis.py mu_cape 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py dcape 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py ebwd 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py lr03 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py lr700_500 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py dp850 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py ta850 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py ta500 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py Umean06 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py s06 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py Umean800_600 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py lr13 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py rhmin13 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py srhe_left 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py q_melting 0 era5
python /home/548/ab4502/working/ExtremeWind/event_analysis.py eff_lcl 0 era5



