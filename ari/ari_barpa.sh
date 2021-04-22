#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=24:00:00,mem=32GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/ari_barpa.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/ari_barpa.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate geopandas

#python working/ExtremeWind/ari/ari.py -m BARPA -y1 1990 -y2 2018
#python working/ExtremeWind/ari/ari.py -m BARPA -y1 2036 -y2 2064
#python working/ExtremeWind/ari/ari.py -m BARPA -y1 2071 -y2 2099
python working/ExtremeWind/ari/ari.py -m BARPA-ERAI -y1 1990 -y2 2015

