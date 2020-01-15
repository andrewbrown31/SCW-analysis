#!/bin/bash

#PBS -P eg3
#PBS -q normal
#PBS -l walltime=48:00:00,mem=64GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_read.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_read.e

conda activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/barra_read.py 2005 2018

