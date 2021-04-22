#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=24:00:00,mem=64GB
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_aws.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_aws.e
#PBS -lstorage=gdata/eg3

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/event_analysis.py logit 0.83 barra is_conv_aws ebwd,lr13,Umean03,ml_el,rhmin03
