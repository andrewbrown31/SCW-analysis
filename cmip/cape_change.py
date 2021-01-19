from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def compute_diff(hist, scenario, months):
	djf = []
	djf_abs = []
	for i in np.arange(len(hist)):
		t2 = np.stack([scenario[i][months[0]].values, scenario[i][months[1]].values, scenario[i][months[2]].values]).mean(axis=0)
		t1 = np.stack([hist[i][months[0]].values, hist[i][months[1]].values, hist[i][months[2]].values]).mean(axis=0)
		djf.append( (t2-t1)/t1 * 100)
		djf_abs.append( (t2-t1) )
	djf_pos = (np.stack(djf) > 0).sum(axis=0)
	djf_neg = (np.stack(djf) <= 0).sum(axis=0)
	djf = np.median(np.stack(djf), axis=0)
	djf_abs = np.median(np.stack(djf_abs), axis=0)
	djf_sig = np.where( ( (djf > 0) & (djf_pos >= 10) ) | ( (djf<=0) & (djf_neg >= 10) ), 1, 0)
	return djf, djf_abs, djf_sig

def load_mean(model, v):
	hist = [xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+m+"_r1i1p1_mean_"+v+"_historical_1979_2005.nc") for m in model]
	scenario = [xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+m+"_r1i1p1_mean_"+v+"_rcp85_2081_2100.nc") for m in model]
	djf, djf_abs, djf_sig = compute_diff(hist, scenario, ["Dec","Jan","Feb"])
	mam, mam_abs, mam_sig = compute_diff(hist, scenario, ["Mar","Apr","May"])
	jja, jja_abs, jja_sig = compute_diff(hist, scenario, ["Jun","Jul","Aug"])
	son, son_abs, son_sig = compute_diff(hist, scenario, ["Sep","Oct","Nov"])
	return [djf, mam, jja, son], [djf_abs, mam_abs, jja_abs, son_abs], [djf_sig, mam_sig, jja_sig, son_sig]

def get_lon_lat(model,v):
	temp = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+model[0]+"_r1i1p1_mean_"+v+"_historical_1979_2005.nc")
	return temp.lon.values, temp.lat.values
		

if __name__ == "__main__":

	model = ["ACCESS1-3", "ACCESS1-0", "BNU-ESM", "CNRM-CM5", "GFDL-CM3", \
			    "GFDL-ESM2G", "GFDL-ESM2M", "IPSL-CM5A-LR", "IPSL-CM5A-MR", \
			    "MIROC5", "MRI-CGCM3", "bcc-csm1-1"]
	lon, lat = get_lon_lat(model, "ml_cape")

	diff_cape, diff_cape_abs, sig_cape = load_mean(model, "ml_cape")
	_, diff_dp850, sig_cape = load_mean(model, "dp850")

	cnt = 0
	plt.figure(figsize=[8,7])
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
		urcrnrlat=-10,projection="cyl")
	s_title = ["DJF","MAM","JJA","SON"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	for i in np.arange(4):
		plt.subplot(2,4,cnt+1)
		c1=plt.contourf(lon, lat, diff_cape[i], vmin=-100, vmax=100, cmap=plt.get_cmap("RdBu_r"))
		plt.contourf(lon,lat,sig_cape[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		plt.title(s_title[i])
		plt.annotate(alph[cnt], xy=(0.05, 0.05), xycoords='axes fraction') 
		
		plt.subplot(2,4,cnt+5)
		c2=plt.contourf(lon, lat, diff_dp850[i], vmin=0, vmax=6, cmap=plt.get_cmap("Reds"))
		plt.contourf(lon,lat,sig_cape[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		plt.title(s_title[i])
		plt.annotate(alph[cnt+4], xy=(0.05, 0.05), xycoords='axes fraction') 
		
		cnt=cnt+1
	plt.subplots_adjust(top=0.9, bottom=0.25, wspace=0.1,left=0.1, hspace=0.3)
	cax1 = plt.axes([0.33,0.63,0.33,0.02])
	cb1=plt.colorbar(c1, cax=cax1, orientation="horizontal", extend="both" )
	cb1.set_label("Mean change (%)")
	cax2 = plt.axes([0.33,0.25,0.33,0.02])
	cb2=plt.colorbar(c2, cax=cax2, orientation="horizontal", extend="both" )
	cb2.set_label("Mean change (J/kg)")
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/cape_change.png", bbox_inches="tight")
