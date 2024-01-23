#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=24:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/cs6_scenario_cmip5_historical_1970_1978.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/cs6_scenario_cmip5_historical_1970_1978.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m ACCESS1-3 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m ACCESS1-0 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m BNU-ESM --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m CNRM-CM5 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m GFDL-CM3 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m GFDL-ESM2G --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m GFDL-ESM2M --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m IPSL-CM5A-LR --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m IPSL-CM5A-MR --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m MIROC5 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m MRI-CGCM3 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m bcc-csm1-1 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m ACCESS-ESM1-5 --force_compute True --scenario_y1 1970 --scenario_y2 1978
python working/ExtremeWind/cmip/cmip_scenario.py -p cs6 -e historical --threshold 33864 -m ACCESS-CM2 --force_compute True --scenario_y1 1970 --scenario_y2 1978
