#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=02:00:00,mem=256GB 
#PBS -l ncpus=32
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/redo_jobs.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/redo_jobs.e 

#Set up conda/shell environments 
conda activate wrfpython3.6
module load mpi4py/3.0.0-py3
module unload python3/3.6.2

#Redo failed days
#2013
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2013122900" "2013122923" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2013123000" "2013123023" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2013123100" "2013123123" "True" "barra_fc" 1

#2014
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2014122800" "2014122823" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2014122900" "2014122923" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2014123000" "2014123023" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2014123100" "2014123123" "True" "barra_fc" 1

#2015
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2015122800" "2015122823" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2015122900" "2015122923" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2015123000" "2015123023" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2015123100" "2015123123" "True" "barra_fc" 1

#2018
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2018010100" "2018010123" "True" "barra_fc" 1
mpiexec python -m mpi4py /home/548/ab4502/working/ExtremeWind/wrf_parallel.py "barra_fc" "aus" "2018123100" "2018123123" "True" "barra_fc" 1

