import xarray as xr
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

if __name__ == "__main__":

	param = np.array(["ml_cape", "mu_cape", "sb_cape", "ml_cin", "sb_cin", "mu_cin",\
				"ml_lcl", "mu_lcl", "sb_lcl", "eff_cape", "eff_cin", "eff_lcl", "dcape", \
				"ml_lfc", "mu_lfc", "eff_lfc", "sb_lfc",\
				"lr01", "lr03", "lr13", "lr36", "lr24", "lr_freezing",\
				"qmean01", "qmean03", "qmean06", "qmean13", "qmean36",\
				"qmeansubcloud", "q_melting", "q1", "q3", "q6",\
				"rhmin01", "rhmin03", "rhmin06", "rhmin13", "rhmin36", \
				"rhminsubcloud", "tei", \
				"mhgt", "mu_el", "ml_el", "sb_el", "eff_el", \
				"pwat", "v_totals", "c_totals", "t_totals", \
				"maxtevv", "te_diff", "dpd850", "dpd700", "pbl_top",\
				\
				"cp", "cape", "dcp2", "cape*s06",\
				\
				"srhe", "srh01", "srh03", "srh06", \
				"srhe_left", "srh01_left", "srh03_left", "srh06_left", \
				"ebwd", "s010", "s06", "s03", "s01", "s13", "s36", "scld", \
				"omega01", "omega03", "omega06", \
				"U500", "U10", "U1", "U3", "U6", \
				"Ust", "Ust_left", "Usr01", "Usr03", "Usr06", "Usr01_left",\
				"Usr03_left", "Usr06_left", \
				"Uwindinf", "Umeanwindinf", "Umean800_600", "Umean06", \
				"Umean01", "Umean03", "wg10",\
				\
				"dcp", "stp_cin", "stp_cin_left", "stp_fixed", "stp_fixed_left",\
				"scp", "ship",\
				"mlcape*s06", "mucape*s06", "sbcape*s06", "effcape*s06", \
				"mlcape*s06_2", "mucape*s06_2", "sbcape*s06_2", \
				"effcape*s06_2", "dmgwind", "hmi", "wmsi_mu", "wmsi_ml",\
				"dmi", "mwpi_mu", "mwpi_ml", "ducs6", "convgust", "windex",\
				"gustex", "gustex2","gustex3", "eff_sherb", "sherb", \
				"moshe", "mosh", "wndg","mburst","sweat","k_index", "esp"])

	drop_param = param[~np.in1d(param,np.array(["k_index","v_totals","t_totals","Umean800_600",\
			"Umean01", "ml_el", "s06", "s03", "dcape"]))]


	from dask.diagnostics import ProgressBar
	ProgressBar().register()

	f = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/erai/erai_*.nc", \
			drop_variables = drop_param, parallel=True) 

	#Conditional parameter
	#sf =  (f["ml_el"]<6000) & (f["wg10"]>=17) & (f["sherb"]>=0.5) & (f["Umean800_600"]>=20)
	#mf = (f["ml_el"]>=6000) & (f["Umean800_600"]>=5) & (f["t_totals"]>=47)
	#Conditional parameter for high lightning events 
	#sf =  (f["ml_el"]<6000) & (f["k_index"]>=20) & (f["s03"]>=10) & (f["Umean800_600"]>=16) & (f["Umean01"]>=10)
	#mf = (f["ml_el"]>=6000) & (f["Umean800_600"]>=5) & (f["t_totals"]>=46) & (f["k_index"]>=30)
	#cond = sf | mf

	#Logistic regression equation
	#TRAINED ON CONV AWS
	#z = f["ml_el"]*6.10366411e-05 + f["k_index"]*1.32055494e-01 + f["t_totals"]*8.23060958e-02 + f["Umean06"]*2.32813508e-01 + f["dcape"]*9.91708172e-04 + f["wg10"]*1.50250086e-01 - 12.30388198
	#p = 1 / ( 1 + np.exp(-z) )
	#cond = ( p >= 0.8 )
	#TRAINED ON CONV AWS with lightning >= 100
	#z = f["ml_el"]*1.24539974e-04 + f["k_index"]*6.49632528e-02 + f["t_totals"]*9.96066930e-02 + f["Umean06"]*2.08557657e-01 + f["dcape"]*8.38427558e-04 - 9.35555394
	#z = f["ml_el"]*3.64413046e-05 + f["k_index"]*1.89036078e-01 + f["t_totals"]*1.55224631e-01 + f["Umean800_600"]*1.48836578e-01 + f["dcape"]*9.90171054e-04 + f["Umean01"]*1.30845911e-01 + f["s06"]*3.05869355e-02 - 16.48218016
	#p = 1 / ( 1 + np.exp(-z) )
	#cond = ( p >= 0.8 )
	#TRAINED SEPARATELY ON SHALLOW AND DEEP CONVECTIVE EVENTS
	deep = f.where(f["ml_el"] >= 6000)
	shallow = f.where(f["ml_el"] < 6000)
	z_deep = deep["k_index"]*0.193268  + \
		deep["t_totals"]*0.127615 + deep["Umean800_600"]*0.153404 + \
		deep["dcape"]*0.001352 + deep["Umean01"]*0.016803 + deep["s06"]*0.057799 - 16.04589324
	z_shallow = shallow["k_index"]*0.208623  + \
		shallow["t_totals"]*0.125546 + shallow["Umean800_600"]*0.132045 + \
		shallow["dcape"]*0.000448 + shallow["Umean01"]*0.234230 + shallow["s03"]*0.029248 - 15.30632583
	p_deep = 1 / ( 1 + np.exp(-z_deep) )
	p_shallow = 1 / ( 1 + np.exp(-z_shallow) )
	cond = np.maximum(p_shallow >= 0.9, p_deep >= 0.6)
	

	#TRAINED ON STA REPORTS
	#z = f["ml_el"]*1.26339649e-04 + f["k_index"]*6.50557294e-02 + f["t_totals"]*1.00580641e-01 + f["Umean06"]*2.00928684e-01 + f["dcape"]*8.34096849e-04 + f["wg10"]*1.89427466e-02 - 9.52532835
	#p = 1 / ( 1 + np.exp(-z) )
	#cond = ( p >= 0.7 )


	from mpl_toolkits.basemap import Basemap
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")

	#Annual grid point frequency
	clim = False
	if clim:
		plt.figure()
		m.drawcoastlines()
		t1 = dt.datetime(1979,1,1)
		t2 = dt.datetime(2017,12,31,18)
		years = (t2.year - t1.year) + 1
		xr.plot.pcolormesh(\
			cond.sel(time=slice(t1, t2))\
			.resample(time="1D").max(dim="time").sum(dim="time") / float(years), \
			x="lon", y="lat",cmap=plt.get_cmap("hot_r",12))

	#Trend
	trend = True
	n_boot = 100
	season = False
	if trend:
		t1 = dt.datetime(1979,1,1)
		t2 = dt.datetime(1998,12,31,18)
		t3 = dt.datetime(1998,1,1)
		t4 = dt.datetime(2017,12,31,18)
		p1 = cond.sel(time=slice(t1, t2)).resample(time="1D").max(dim="time")
		p2 = cond.sel(time=slice(t3, t4)).resample(time="1D").max(dim="time")
		plt.figure()
		m.drawcoastlines()
		xr.plot.pcolormesh( ((p2.sum(dim="time") - p1.sum(dim="time")) / p1.sum(dim="time")) * 100, \
			x="lon", y="lat",cmap=plt.get_cmap("bwr",12), robust=True)

		if n_boot >= 0:
			from event_analysis import hypothesis_test
			x,y = np.meshgrid(cond.lon, cond.lat)
			sig = hypothesis_test(p1.to_masked_array(), p2.to_masked_array(), n_boot)
			plt.plot(x[np.where(sig<=0.05)], y[np.where(sig<=0.05)], "ko", markersize=1)

		if season:

			#FOR A SEASON:
			season = [12,1,2]
			plt.figure()
			m.drawcoastlines()
			xr.plot.pcolormesh(\
				( cond[np.in1d(cond["time.month"], season)].sel(time=slice(t3, t4))\
				.resample(time="1D").max(dim="time").sum(dim="time") )\
				- ( cond[np.in1d(cond["time.month"], season)].sel(time=slice(t1, t2))\
				.resample(time="1D").max(dim="time").sum(dim="time") ), \
				x="lon", y="lat",cmap=plt.get_cmap("bwr"))
		
	#Ingredients
	ingreds = False
	p = "v_totals"
	if ingreds:
		plt.figure()
		plt.subplot(131)
		m.drawcoastlines()
		f[p].load().quantile(.5, dim="time", keep_attrs=True).\
			plot(cmap=plt.get_cmap("YlGnBu"), x="lon", y="lat")
		plt.subplot(132)
		m.drawcoastlines()
		f[p].load().quantile(.9, dim="time", keep_attrs=True).\
			plot(cmap=plt.get_cmap("YlGnBu"), x="lon", y="lat")
		plt.subplot(133)
		m.drawcoastlines()
		f[p].load().quantile(.99, dim="time", keep_attrs=True).\
			plot(cmap=plt.get_cmap("YlGnBu"), x="lon", y="lat")

		t1 = dt.datetime(1979,1,1)
		t2 = dt.datetime(1998,12,31,18)
		t3 = dt.datetime(1998,1,1)
		t4 = dt.datetime(2017,12,31,18)
		p1 = f[p].load().sel(time=slice(t1, t2))
		p2 = f[p].load().sel(time=slice(t3, t4))

		plt.figure()
		plt.subplot(131)
		m.drawcoastlines()
		( ( ( (p2.quantile(.5, dim="time", keep_attrs=True) ) - \
			(p1.quantile(.5, dim="time", keep_attrs=True) ) ) /\
			(p1.quantile(.5, dim="time", keep_attrs=True) ) ) *100) .\
			plot(cmap=plt.get_cmap("bwr"), x="lon", y="lat")
		plt.subplot(132)
		m.drawcoastlines()
		( ( ( (p2.quantile(.9, dim="time", keep_attrs=True) ) - \
			(p1.quantile(.9, dim="time", keep_attrs=True) ) ) /\
			(p1.quantile(.9, dim="time", keep_attrs=True) ) ) *100) .\
			plot(cmap=plt.get_cmap("bwr"), x="lon", y="lat")
		plt.subplot(133)
		m.drawcoastlines()
		( ( ( (p2.quantile(.99, dim="time", keep_attrs=True) ) - \
			(p1.quantile(.99, dim="time", keep_attrs=True) ) ) /\
			(p1.quantile(.99, dim="time", keep_attrs=True) ) )*100)  .\
			plot(cmap=plt.get_cmap("bwr"), x="lon", y="lat")



	plt.show()

