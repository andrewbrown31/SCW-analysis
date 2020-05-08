#!/bin/bash

#For the wrfpython3.6 environment
conda activate wrfpython3.6
cd /g/data/eg3/ab4502/wrf-python
pip uninstall wrf-python
cd build_scripts
sh gnu_omp.sh
cp ~/specialdec.py /g/data/eg3/ab4502/miniconda3/envs/wrfpython3.6/lib/python3.6/site-packages/wrf/specialdec.py
cp ~/extension.py /g/data/eg3/ab4502/miniconda3/envs/wrfpython3.6/lib/python3.6/site-packages/wrf/extension.py
cp ~/metadecorators.py /g/data/eg3/ab4502/miniconda3/envs/wrfpython3.6/lib/python3.6/site-packages/wrf/metadecorators.py

