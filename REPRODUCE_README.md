This text file contains a basic reconstructed log of the functions and software called to produce the 
results in Brown and Dowdy (2020) doi:

This log utilises python scripts written by the authors, housed in this git repository. This is 
 a working repository, and so it is not guaranteed that the software appears exactly as was used
 to generate the abovementioned paper

#Set up python environment using conda
1> conda create --name wrfpython3.6 --file requirements.txt
2> conda activate wrfpython3.6
3> sh wrf-python/compile_wrf_python.sh

#Data processing
4> sh jobs/era5_wrfpython/wrfpython_parallel_era5_driver.sh
5> sh jobs/era5_points/era5_points_driver.sh
6> sh jobs/barra_wrfpython/wrfpython_parallel_barra_driver.sh
7> sh jobs/barra_points/barra_points_driver.sh
8> ipython
	9> join the yearly points data which was output from lines 2,4 using pandas (python package).
	   This was done interactively, without a script.
10> ipython
	11> from obs_read import read_convective_wind_gusts, read_upperair_obs
	12> read_convective_wind_gusts()
	13> read_upperair_obs(dt.datetime(2005,1,1),dt.datetime(2018,12,31),"UA_wrfpython", "wrfpython")

14> ipython
	15> from event_analysis import optimise_pss
	16> optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="era5",time="floor")
	17> optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="era5",time="ceil")
	18> optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="barra_fc",time="ceil")
	19> optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl",T=1000, compute=True, l_thresh=2, is_pss="hss", model_name="barra_fc",time="floor")

# Logistic model testing (Table 2)
20> ipython
	21> from logit import logit_test
	22> logit_test(["ml_cape","ml_el","srhe_left","Umean06","s06","lr_freezing","qmean01","qmeansubcloud","lr36","mhgt"],"barra","is_conv_aws")
	23> logit_test(["ml_cape","srhe_left","ml_el","Umean06","s03","lr36","lr_freezing","mhgt","qmeansubcloud","qmean01"],"era5","is_conv_aws")

#Figure 1
24> ipython
	25> from plot_param import sta_versus_aws
	26> sta_versus_aws()

#Figure 2, 4 and 5
27> ipython
	28> from plot_param import plot_ranked_hss
	29> plot_ranked_hss()

#Figure 3
30> ipython
	31> from event_analysis import diagnostics_aws_compare
	32> diagnostics_aws_compare()

#Figure 6
33> ipython
	34> from plot_param import plot_candidate_variable_kde
	35> plot_candidate_variable_kde()

#Figure 7
36> ipython 
	37> from plot_param import obs_versus_model
	38> obs_versus_model("BARRA")
	39> obs_versus_model("ERA5")

#Figure A1
40> ipython
	41> from event_analysis import compare_obs_soundings
	42> compare_obs_soundings()
