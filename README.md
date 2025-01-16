# Calculating SCW diagnostics
This is a clean branch containing the minimum amount of code needed to compute a suite of convective diagnostics, as used by Brown and Dowdy (2021). This code relies heavily on [wrf-python](https://wrf-python.readthedocs.io/en/latest/), [SkewT](https://github.com/tjlang/SkewT), and [metpy](https://unidata.github.io/MetPy/latest/index.html).

Below shows how to set up the python environment, and how to run an example computation on ERA5 data.

## Installation
The recommended installation is to use [conda](https://conda.io/projects/conda/en/latest/index.html). However, we also need to compile wrf-python from source, as we will make some changes to the source code. This requires a fortran compiler.

```bash
#Download the code
git clone -b clean https://github.com/andrewbrown31/SCW-analysis.git
cd SCW-analysis

#Create the conda environment (may take some time to solve)
conda create --name wrfpython_test -c conda-forge python=3.6 metpy=0.10.2 wrf-python=1.3.2 netcdf4=1.5.1.2 pint=0.9 cdsapi=0.5.1
conda activate wrfpython_test

#Download the source code for wrf-python, and unpack
wget https://files.pythonhosted.org/packages/72/ad/e836d18cd1b06c58a2f22559700f23c45e00ca4c36c23b4ffe2a35e718c7/wrf-python-1.3.1.tar.gz
tar -xvf wrf-python-1.3.1.tar.gz
rm wrf-python-1.3.1.tar.gz

#From our github repo, copy an edited version of rip_cape.f90 fortran code to the wrf-python source directory, as well as some manually edited wrapper functions
cd wrf-python-1.3.1/
cp ../wrf-python/rip_cape.f90 fortran/rip_cape.f90
cp ../wrf-python/specialdec.py src/wrf/specialdec.py
cp ../wrf-python/extension.py src/wrf/extension.py
cp ../wrf-python/metadecorators.py src/wrf/metadecorators.py

#Build the wrf-python code. Need to first uninstall the existing wrf-python package from the conda env. This requires a fortran compiler
pip uninstall wrf-python
cd build_scripts/
sh gnu_omp.sh
```

## Example

An example is included here in `example.py` based on ERA5 data in the `read_model_data/data/` directory, downloaded from the ECMWF CDS as shown below (this requires an [API key from CDS](https://cds.climate.copernicus.eu/api-how-to)).
`example.py` also has the option to use data from the rt52 project on Gadi, or BARRA data from cj37.

Output from this example is saved as `read_model_data/data/era5_20160928_20160928.nc`.
```bash
#(Optional) Download some ERA5 data for example.py
#cd ../../read_model_data/
#python download_sfc.py
#python download_pl.py
#cd ..

#Run the diagnostic code.
python example.py
```

## Citing
If you'd like to aknowledge the use of this code in a publication, you can cite it with [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14433710.svg)](https://doi.org/10.5281/zenodo.14433710)

## References
Brown, A., & Dowdy, A., (2021a) Severe convection-related winds in Australia and their associated environments. Journal of Southern Hemisphere Earth Systems Science, -. https://doi.org/10.1071/ES19052
