from event_analysis import optimise_pss, pss
import multiprocessing
import pandas as pd
import numpy as np
import itertools

def test_csi(df, pss_df, event, is_pss, param_list, l_thresh):

	df[event] = np.nan
	if (event == "is_high_U"):
		df.loc[(df["is_conv_aws"]==1) & (df["Umean800_600"]>=20),event]=1
		df.loc[(df["is_conv_aws"]==0) & (df["Umean800_600"]<20),event]=0
	if (event == "is_low_el"):
		#Given a high ML-EL and lightning (deep convective case), what variable indicates a SCW event
		df.loc[(df["lightning"]>=l_thresh) & (df["wind_gust"]>=25) & (df["ml_el"]<6000),event]=1
		df.loc[(df["lightning"]>=l_thresh) & (df["wind_gust"]<25) & (df["ml_el"]<6000),event]=0
	if (event == "is_high_el"):
		#Given a high ML-EL and lightning (deep convective case), what variable indicates a SCW event
		df.loc[(df["lightning"]>=l_thresh) & (df["wind_gust"]>=25) & (df["ml_el"]>=6000),event]=1
		df.loc[(df["lightning"]>=l_thresh) & (df["wind_gust"]<25) & (df["ml_el"]>=6000),event]=0
	if (event == "is_high_el_lightning"):
		#Given a low ML-EL (shallow convective case), what variable indicates lightning
		df.loc[(df["lightning"]>=l_thresh) & (df["ml_el"]>=6000),event]=1
		df.loc[(df["lightning"]<l_thresh) & (df["ml_el"]>=6000),event]=0
	if (event == "is_low_el_lightning"):
		#Given a low ML-EL (shallow convective case), what variable indicates lightning
		df.loc[(df["lightning"]>=l_thresh) & (df["ml_el"]<6000),event]=1
		df.loc[(df["lightning"]<l_thresh) & (df["ml_el"]<6000),event]=0

	for p in param_list:
		print(p)
		test_thresh = np.linspace(df.loc[:,p].min(), np.percentile(df.loc[:,p],99.95) , T)
		temp_df = df.loc[:,[event,p]]
		iterable = itertools.product(test_thresh, [temp_df], [p], [event], [is_pss])
		res = pool.map(pss, iterable)
		thresh = [res[i][1] for i in np.arange(len(res))]
		pss_p = [res[i][0] for i in np.arange(len(res))]

		pss_df.loc[p, ("threshold_"+event)] = thresh[np.argmax(np.array(pss_p))]
		pss_df.loc[p, ("pss_"+event)] = np.array(pss_p).max()

	return df, pss_df

if __name__ == "__main__":

	T = 1000
	is_pss = "pss"
	l_thresh = 100

	#----------------------------------------------------------------------------------------------------------

	model_fname = "/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl"
	model = pd.read_pickle(model_fname).set_index(["time", "loc_id"])
	param_list = np.delete(np.array(model.columns), \
			np.where((np.array(model.columns)=="lat") | (np.array(model.columns)=="lon")))
	pool = multiprocessing.Pool()
	pss_df, df = optimise_pss(model_fname, compute=False, l_thresh=l_thresh, is_pss=is_pss)

	df, pss_df = test_csi(df, pss_df, "is_high_el_lightning", is_pss, param_list, l_thresh)
	df, pss_df = test_csi(df, pss_df, "is_low_el_lightning", is_pss, param_list, l_thresh)
	df, pss_df = test_csi(df, pss_df, "is_high_el", is_pss, param_list, l_thresh)
	df, pss_df = test_csi(df, pss_df, "is_low_el", is_pss, param_list, l_thresh)


