#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=10:00:00,mem=128GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario_2006_2018.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario_2006_2018.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m MRI-CGCM3 --force_compute True --scenario_y1 2006 --scenario_y2 2018

