#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=12:00:00,mem=128GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/logit_aws_scenario.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6 

python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e historical --threshold 0.72 -m ERA5 --force_compute True
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m ACCESS1-3
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m ACCESS1-0
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m BNU-ESM --force_compute True
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m CNRM-CM5
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m GFDL-CM3
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m GFDL-ESM2G
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m GFDL-ESM2M
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m IPSL-CM5A-LR
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m IPSL-CM5A-MR
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m MIROC5
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m MRI-CGCM3
#python working/ExtremeWind/cmip/cmip_scenario.py -p logit_aws -e rcp85 --threshold 0.72 -m bcc-csm1-1

