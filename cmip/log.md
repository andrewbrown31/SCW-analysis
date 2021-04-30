# Log for reproducability

This text file contains a basic reconstructed log of the functions and software called to produce the 
results in Brown and Dowdy, Severe Convective Wind Environments and Future Projected Changes in Australia (in prep.) JGR-Atmopheres

This is a working repository, and so it is not guaranteed that the scripts appear exactly as was used
to generate the abovementioned paper. Although previous states can be accessed through navagating commits.
 
#### Set up main python environment for convective diagnostics using conda
```bash
conda create --name wrfpython3.6 --file ../requirements.txt
conda activate wrfpython3.6
sh ../wrf-python/compile_wrf_python.sh
```

#### Set up geopandas python environment for using NRM regions
```bash
conda create --name geopandas --file requirements_geopandas.txt
```

#### Create map of stations and ERA5 topography
[stn_map.ipynb](https://github.com/andrewbrown31/SCW-analysis/blob/master/stn_map.ipynb)

#### Run convective diagnostic code on ERA5, ERA5 and MERRA-2 and extract point data at 35 station locations (4 points only for MERRA-2 and BARRA)
```bash
sh ../jobs/era5_wrfpython/wrfpython_parallel_era5_driver.sh
sh ../jobs/era5_points/era5_points_driver.sh
sh ../jobs/barra_wrfpython/wrfpython_parallel_barra_driver.sh
sh ../jobs/barra_points/barra_points_driver.sh
sh ../jobs/merra_wrfpython/wrfpython_nonparallel_merra_driver.sh
sh ../jobs/merra2_points/merra2_points_driver.sh
```

#### Create observed SCW event datasets, calculate convective diagnostics from radiosondes, and compute skill scores from ERA5 (Table 1)
```python
from SCW-analysis/obs_read import read_convective_wind_gusts, read_upperair_obs
read_convective_wind_gusts()
read_upperair_obs(dt.datetime(2005,1,1),dt.datetime(2018,12,31),"UA_wrfpython", "wrfpython")
```
```bash
conda activate wrfpython3.6
python skill.py
```

#### Create figure of ERA5 diagnostic  variability compared with events
```bash
conda activate wrfpython3.6
python era5_variability.py
```

#### From ERA5 and MERRA-2 convective diagnostics, create monthly gridded data of threshold exceedences as well as monthly means
```bash
sh ../jobs/create_threshold_clim/era5_vars.sh
sh ../jobs/create_threshold_clim/merra_vars.sh
```

#### From monthly data, resample to annual and compute spatial trends (Figure 3)
```bash
conda activate wrfpython3.6
python spatial_hist_trends_era5.py
```

#### Compute diagnostics on 12 GCM models for the historical period (1979-2005)
```bash
sh ACCESS1-0_historical_r1i1p1.sh
sh ACCESS1-3_historical_r1i1p1.sh
sh BNU-ESM_historical_r1i1p1.sh
sh CNRM-CM5_historical_r1i1p1.sh
sh GFDL-CM3_historical_r1i1p1.sh
sh GFDL-ESM2G_historical_r1i1p1.sh
sh GFDL-ESM2M_historical_r1i1p1.sh
sh IPSL-CM5A-LR_historical_r1i1p1.sh
sh IPSL-CM5A-MR_historical_r1i1p1.sh
sh MIROC5_historical_r1i1p1.sh
sh MRI-CGCM3_historical_r1i1p1.sh
sh bcc-csm1-1_historical_r1i1p1.sh
```

#### Compute diagnostics on 12 GCM models for the future period (2081-2100)
```bash
sh ACCESS1-0_rcp85_r1i1p1.sh
sh ACCESS1-3_rcp85_r1i1p1.sh
sh BNU-ESM_rcp85_r1i1p1.sh
sh CNRM-CM5_rcp85_r1i1p1.sh
sh GFDL-CM3_rcp85_r1i1p1.sh
sh GFDL-ESM2G_rcp85_r1i1p1.sh
sh GFDL-ESM2M_rcp85_r1i1p1.sh
sh IPSL-CM5A-LR_rcp85_r1i1p1.sh
sh IPSL-CM5A-MR_rcp85_r1i1p1.sh
sh MIROC5_rcp85_r1i1p1.sh
sh MRI-CGCM3_rcp85_r1i1p1.sh
sh bcc-csm1-1_rcp85_r1i1p1.sh
```

#### Apply quantile-mapping to CMIP5 data using ERA5, and save monthly and seasonal frequency/means for each diagnostic, for historical and future period 
```bash
sh jobs/cmip5_rcp85_2081_2100/*.sh
```

#### Calculate projected changes for Australia (Figure 4) and each NRM region (Figure 5), for four diagnostics, as well as for instability variables (Figure 6)
```bash 
conda activate geopandas
python percent_mean_change_indices.py
conda activate wrfpython3.6
python ttotals_components.py
```

#### Supplementry/other material
[Figure S1](https://github.com/andrewbrown31/SCW-analysis/blob/master/reanalysis_distr_compare.ipynb)
```bash
#Figure S2 and S3
conda activate wrfpython3.6
python save_cmip_mean.py
python variable_compare.py

#Figure S4 and S5
python spatial_ingredient_mean_trends_era5.py

#Figure S6
python spatial_hist_trends_era5_merra2.py

#Figure S7
python plot_seasonal_freq.py

#Figure S8
python logit_components.py
```
[Table S4](https://github.com/andrewbrown31/SCW-analysis/cmip/era5_nrm_trend.ipynb)
