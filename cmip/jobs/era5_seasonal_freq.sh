#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=24:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_seasonal_freq_scenario.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_seasonal_freq_scenario.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

#Seeing as the QM scripts for DCP, SCP, CS6 and logit indices do not load
# ERA5 data and save the seasonal frequency, do it here instead

#python working/ExtremeWind/cmip/cmip_scenario.py -p t_totals -m ERA5 --threshold 50.9 -e historical --hist_only True
#python working/ExtremeWind/cmip/cmip_scenario.py -p dcp -m ERA5 --threshold 0.91 -e historical --hist_only True
#python working/ExtremeWind/cmip/cmip_scenario.py -p eff_sherb -m ERA5 --threshold 0.90 -e historical --hist_only True
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e historical --threshold 0.83 -m ERA5 --hist_only True --force_compute True
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_sta -e historical --threshold 0.83 -m ERA5 --hist_only True --force_compute True

