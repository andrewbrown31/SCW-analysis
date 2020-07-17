#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=06:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario_cmip6_ssp585_2081_2100.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario_cmip6_ssp585_2081_2100.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e ssp585 --threshold 0.72 -m ACCESS-CM2 --force_compute True --scenario_y1 2081 --scenario_y2 2100
