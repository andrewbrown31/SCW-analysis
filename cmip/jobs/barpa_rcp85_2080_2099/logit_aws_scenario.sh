#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=36:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_rcp85_2081_2100.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_rcp85_2081_2100.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p lr36 -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
python working/ExtremeWind/cmip/cmip_scenario.py -p mhgt -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
python working/ExtremeWind/cmip/cmip_scenario.py -p ml_el -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
python working/ExtremeWind/cmip/cmip_scenario.py -p qmean01 -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
python working/ExtremeWind/cmip/cmip_scenario.py -p srhe_left -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
python working/ExtremeWind/cmip/cmip_scenario.py -p Umean06 -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws --threshold 0.72 -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws_barra --threshold 0.80 -e rcp85 -m BARPA --force_compute True --scenario_y1 2080 --scenario_y2 2099
