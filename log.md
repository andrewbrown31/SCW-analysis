# Log for reproducability

This text file contains a basic reconstructed log of the functions and software called to produce the 
results in Brown and Dowdy (in prep.)

 This is a working repository, and so it is not guaranteed that the scripts appear exactly as was used
 to generate the abovementioned paper. Although previous states can be accessed through navagating commits.

#### Set up python environment using conda
```bash
conda create --name wrfpython3.6 --file requirements.txt
conda activate wrfpython3.6
sh wrf-python/compile_wrf_python.sh
```

#### Calculating convective diagnostics from reanalyses
```bash
sh jobs/era5_wrfpython/wrfpython_parallel_era5_driver.sh
sh jobs/era5_points/era5_points_driver.sh
sh jobs/barra_wrfpython/wrfpython_parallel_barra_driver.sh
sh jobs/barra_points/barra_points_driver.sh
```

#### Create observed SCW event datasets, calculate convective diagnostics from radiosondes, and compute skill scores from reanalysis
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
#### Logistic regression model selection and evaluation (Table 2, 3, A2, A3)
```python
from logit import fwd_selection, logit_explicit_cv, colin_test
from event_analysis import auc_test
fwd_selection("era5", "is_sta", False)
fwd_selection("barra", "is_conv_aws", False)
fwd_selection("barra", "is_sta", False)
fwd_selection("era5", "is_conv_aws", False)
#Get cross-validated skill for each regression model...
logit_explicit_cv("logit_cv_skill.csv")
#Get the AUC and confidence intervals for each variable selected...
auc_test()
#Check co-linearity...
colin_test()
```
#### Figures
```python
from plot_param import sta_versus_aws, plot_ranked_hss, plot_candidate_variable_kde, plot_candidate_kde_logit_exclude, obs_versus_mod_plot, hss_function, plot_ranked_hss_proximity_test, plot_ranked_hss_proximity_test_barra, plot_candidate_variable_kde_sta
from event_analysis import diagnostics_aws_compare, compare_obs_soundings
from logit import plot_roc

sta_versus_aws()                       #Fig. 1
plot_ranked_hss()                      #Fig. 2
diagnostics_aws_compare()              #Fig. 3
plot_candidate_variable_kde()          #Fig. 4
obs_versus_mod_plot()                  #Fig. 5
compare_obs_soundings()                #Fig. A1
plot_ranked_hss_proximity_test()       #Fig. A2
plot_ranked_hss_proximity_test_barra() #Fig. A3
hss_function()                         #Fig. A4
plot_roc()                             #Fig. A5
plot_candidate_variable_kde_sta()      #Fig. A6
```
