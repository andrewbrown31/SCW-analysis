#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=24:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/ari_barra-ad.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/ari_barra-ad.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate geopandas

python working/ExtremeWind/ari/ari.py -m BARRA-AD -y1 1990 -y2 2018

