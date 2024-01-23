import numpy as np
import xarray as xr
import glob
import tqdm
from GadiClient import GadiClient

def calc_bdsd(ds):
    ds = ds.assign(bdsd = 1 /
                   ( 1 + np.exp( -(6.1e-02*ds["ebwd"] + 1.5e-01*ds["Umean800_600"] + 9.4e-01*ds["lr13"] + 3.9e-02*ds["rhmin13"] +
                                   1.7e-02*ds["srhe_left"] +3.8e-01*ds["q_melting"] +4.7e-04*ds["eff_lcl"] - 1.3e+01 ) ) ) )
    return ds

if __name__ == "__main__":

    client = GadiClient()

    for y in np.arange(2005,2016):
        print(y)
        files = np.sort(glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5_"+str(y)+"*"))
        for f in tqdm.tqdm(files):
            era5 = xr.open_dataset(f,chunks={"time":24})
            calc_bdsd(era5).bdsd.mean("time").to_netcdf(f.replace("era5_","bdsd_avg_"))
            era5.close()
