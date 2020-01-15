#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=12:00:00,mem=32GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/logit.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/logit.e

conda activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/logit.py

