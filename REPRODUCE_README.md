# Log for reproducability

This text file contains a basic reconstructed log of the functions and software called to produce the 
results in Brown and Dowdy (2020) doi:

This log utilises python scripts written by the authors, housed in this git repository. This is 
 a working repository, and so it is not guaranteed that the software appears exactly as was used
 to generate the abovementioned paper. Although previous states can be accessed through navagating commits.

#### Set up python environment using conda
```bash
conda create --name wrfpython3.6 --file requirements.txt
conda activate wrfpython3.6
sh wrf-python/compile_wrf_python.sh
```

#### Data processing
```bash
sh jobs/era5_wrfpython/wrfpython_parallel_era5_driver.sh
sh jobs/era5_points/era5_points_driver.sh
sh jobs/barra_wrfpython/wrfpython_parallel_barra_driver.sh
sh jobs/barra_points/barra_points_driver.sh
```

#### In python (after concaternating point output using pandas)
```python
from obs_read import read_convective_wind_gusts, read_upperair_obs
read_convective_wind_gusts()
read_upperair_obs(dt.datetime(2005,1,1),dt.datetime(2018,12,31),"UA_wrfpython", "wrfpython")
```
```python
from event_analysis import optimise_pss
optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="era5",time="floor")
optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="era5",time="ceil")
optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="barra_fc",time="ceil")
optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="barra_fc",time="floor")
```
#### Logistic model testing (Table 2)
```bash
sh jobs/logit_selection/logit_barra_aws.sh
sh jobs/logit_selection/logit_barra_sta.sh
sh jobs/logit_selection/logit_era5_aws.sh
sh jobs/logit_selection/logit_era5_sta.sh
```
#### Figures
```python
from plot_param import sta_versus_aws, plot_ranked_hss, plot_candidate_variable_kde, plot_candidate_kde_logit_exclude, obs_versus_mod_plot
from event_analysis import diagnostics_aws_compare, compare_obs_soundings

sta_versus_aws()                    #Fig. 1
plot_ranked_hss()                   #Fig. 2, 4, 5
diagnostics_aws_compare()           #Fig. 3
plot_candidate_variable_kde()       #Fig. 6
obs_versus_mod_plot()               #Fig. 7
compare_obs_soundings()             #Fig. A1
plot_candidate_kde_logit_exclude()  #Fig. A2
```
