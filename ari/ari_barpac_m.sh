#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=24:00:00,mem=64GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/ari_barpac_m.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/ari_barpac_m.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38+gdata/tp28

source activate geopandas

python working/ExtremeWind/ari/ari.py -m BARPAC-M-ACCESS1-0 -y1 1985 -y2 2005
python working/ExtremeWind/ari/ari.py -m BARPAC-M-ACCESS1-0 -y1 2039 -y2 2058

