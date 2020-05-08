from event_analysis import optimise_pss, pss
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import multiprocessing
import itertools
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

def logit_test(all_predictors, model, event, model_diagnostics=None):

	#Try every possible combination of "all_predictors" predictors, and 
	#	train a logstic model using cross validation
	#10 variables = 1023 combinations

	#Load diagnostics/events
	if model == "era5":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"era5_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5")
	elif model == "barra":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra")
	else:
		raise ValueError("Invalid model name")

	#Set the correct dataframe based on event type 
	if event=="is_sta":
		df = df_sta
	elif event=="is_conv_aws":
		df = df_aws
	
	#Create 16 unique train/test datasets with balanced number of events
	train_dfs = []
	test_dfs = []
	split = StratifiedShuffleSplit(n_splits=16,test_size=0.8,random_state=0)
	for test_index, train_index in split.split(X=df, y=df[event]):
		train_dfs.append(df.iloc[train_index,:])
		test_dfs.append(df.iloc[test_index,:])
			
	#For each combination of variables, calculate the optimal pss/hss, and save the 
	# scores and thresholds
	param_out = []
	hss_thresh_out = []
	pss_thresh_out = []
	hss_out = []
	pss_out = []
	for r in np.arange(2,11):
		params = itertools.combinations(all_predictors, r)
		for predictors in params:
			pool = multiprocessing.Pool()
			print(predictors)
			start = dt.datetime.now()

			iterable = itertools.product(np.arange(16),\
				[train_dfs], [test_dfs], [predictors], [event])
			res = pool.map(logit_train, iterable)

			param_out.append(" ".join(predictors))
			hss_out.append(np.mean([res[i][0] for i in np.arange(16)]))
			pss_out.append(np.mean([res[i][1] for i in np.arange(16)]))
			hss_thresh_out.append(np.mean([res[i][2] for i in np.arange(16)]))
			pss_thresh_out.append(np.mean([res[i][3] for i in np.arange(16)]))
			pool.close()
			pool.join()

	pd.DataFrame({"predictors":param_out, "hss":hss_out, "pss":pss_out,\
		"pss_thresh":pss_thresh_out, "hss_thresh":hss_thresh_out}).\
		to_csv("/g/data/eg3/ab4502/ExtremeWind/points/logit_skill_"+model+"_"+event+".csv",\
		index=False)	

	#Now do the same scores for diagnostics
	try:
		for p in model_diagnostics:
			print(p)
			hss_thresh_out = []
			hss_out = []
			for i in np.arange(len(test_dfs)):
				test_thresh = np.linspace(df.loc[:,p].min(), \
					np.percentile(df.loc[:,p],99.95), 1000)
				iterable = itertools.product(test_thresh, [test_dfs[i]], [p],\
					[event], ["hss"])
				res = pool.map(pss, iterable)
				thresh = [res[i][1] for i in np.arange(len(res))]
				pss_p = [res[i][0] for i in np.arange(len(res))]
				
				hss_out.append(np.max(pss_p))
				hss_thresh_out.append(thresh[np.argmax(np.array(pss_p))])
			pd.DataFrame({"predictors":[p], "hss":[np.mean(hss_out)], \
				"hss_thresh":[np.mean(hss_thresh_out)]}).\
				to_csv("/g/data/eg3/ab4502/ExtremeWind/points/"+p+\
					"_skill_"+model+"_"+event+".csv",\
				index=False)	
	except:
		pass


def logit_train(it):

	i, df_train, df_test, predictors, event = it
	logit = LogisticRegression(class_weight="balanced", solver="liblinear",\
		max_iter=1000)
	logit_mod = logit.fit(df_train[i][list(predictors)], df_train[i][event])
	p = logit_mod.predict_proba(df_test[i][list(predictors)])[:,1]
	df_test_temp = df_test[i]
	df_test_temp.loc[:, "logit"] = p
	#Calculate the HSS/PSS for a range of probabilistic thresholds and save the maximums
	hss_out = []
	pss_out = []
	thresh_hss_out = []
	thresh_pss_out = []
	for t in np.arange(0,1.01,0.01):
		hss_p, thresh_hss = pss([t, df_test_temp, "logit", event, "hss"])
		pss_p, thresh_pss = pss([t, df_test_temp, "logit", event, "pss"])
		hss_out.append(hss_p)
		pss_out.append(pss_p)
		thresh_hss_out.append(thresh_hss)
		thresh_pss_out.append(thresh_pss)

	return [np.max(hss_out), np.max(pss_out), thresh_hss_out[np.argmax(hss_out)], \
			thresh_pss_out[np.argmax(pss_out)] ]


def run_logit(it):
	i, df, predictors, predictors_logit, normalised, test_cond, test_param = it
	if normalised:
		X_train, X_test, y_train, y_test = train_test_split(\
			(df[predictors] - (df[predictors].mean())) / (df[predictors].std()), df["is_conv_aws"])
		test_cond = False
	else:
		X_train, X_test, y_train, y_test = train_test_split(df[predictors], df["is_conv_aws"],random_state=i)

	logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
	logit_mod = logit.fit(X_train[predictors_logit], y_train)
	preds = logit_mod.predict_proba(X_test[predictors_logit])
	preds = (preds[:,1]>=0.6)*1

	hits = ((preds==1) & (y_test==1)).sum()
	misses = ((preds==0) & (y_test==1)).sum()
	fa = ((preds==1) & (y_test==0)).sum()
	cn = ((preds==0) & (y_test==0)).sum()
	#csi_logit = ( hits / (hits+misses+fa) ) 				#CSI
	csi_logit = ( 2*(hits*cn - misses*fa) ) / \
		( misses^2 + fa^2 + 2*hits*cn + (misses + fa) * (hits + cn) ) 	#HSS
	pod_logit = ( hits / (hits+misses) )
	far_logit = (fa / (hits + fa) )

	if test_cond:

		preds_cond = ( (X_test["ml_el"]>=6000) & (X_test["Umean800_600"]>=5) & (X_test["t_totals"]>=46) \
			& (df["k_index"]>=30) ) | \
			( (X_test["ml_el"]<6000) & (X_test["Umean800_600"]>=16) & (X_test["k_index"]>=20) & \
				(X_test["Umean01"]>=10) & (df["s03"]>=10) )
		hits = ((preds_cond==1) & (y_test==1)).sum()
		misses = ((preds_cond==0) & (y_test==1)).sum()
		fa = ((preds_cond==1) & (y_test==0)).sum()
		cn = ((preds_cond==0) & (y_test==0)).sum()
		#csi_cond = ( hits / (hits+misses+fa) ) 					#CSI
		csi_cond = ( 2*(hits*cn - misses*fa) ) / \
			( misses^2 + fa^2 + 2*hits*cn + (misses + fa) * (hits + cn) ) 		#HSS
		pod_cond = ( hits / (hits+misses) )
		far_cond = (fa / (hits + fa) )

		if test_param:

			hits = ((X_test.loc[:,"t_totals"]>=47.4) & (y_test==1)).sum()
			misses = ((X_test.loc[:,"t_totals"]<47.4) & (y_test==1)).sum()
			fa = ((X_test.loc[:,"t_totals"]>=47.4) & (y_test==0)).sum()
			cn = ((X_test.loc[:,"t_totals"]<47.4) & (y_test==0)).sum()
			csi_param = ( hits / (hits+misses+fa) )
			pod_param = ( hits / (hits+misses) )
			far_param = (fa / (hits + fa) )
		
			return [csi_logit, pod_logit, far_logit, csi_cond, pod_cond, far_cond \
					,csi_param, pod_param, far_param]
	
		else:

			return [csi_logit, pod_logit, far_logit, csi_cond, pod_cond, far_logit]

	else:

		return [csi_logit, pod_logit, far_logit, np.nan, np.nan]

def train_logit_cv(model, event, predictors, predictors_logit, N):

	#Given a list of predictors (predictors logit), train a logistic regression model N times
	# with cross-validation, and output test stats
	#Also outputs conditional and threshold test stats (e.g. using total totals as a threshold),
	#	although the details of these tests must be changed in run_logit()

	import matplotlib.pyplot as plt

	pool = multiprocessing.Pool()

	if model == "era5":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"era5_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5")
	elif model == "barra":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra")
	else:
		raise ValueError("Invalid model name")
	
	test_cond = True
	test_param = True
	normalised = False
	if event=="is_sta":
		iterable = itertools.product(np.arange(0,N), \
			[df_sta[np.append(predictors,event)]], \
			[predictors], [predictors_logit], [normalised], [test_cond], [test_param])
	elif event=="is_conv_aws":
		iterable = itertools.product(np.arange(0,N), \
			[df_aws[np.append(predictors,event)]], \
			[predictors], [predictors_logit], [normalised], [test_cond], [test_param])
	print("Training Logit...")
	res = pool.map(run_logit, iterable)

	csi_logit = [res[i][0] for i in np.arange(0,len(res))]
	pod_logit = [res[i][1] for i in np.arange(0,len(res))]
	far_logit = [res[i][2] for i in np.arange(0,len(res))]
	csi_cond = [res[i][3] for i in np.arange(0,len(res))]
	pod_cond = [res[i][4] for i in np.arange(0,len(res))]
	far_cond = [res[i][5] for i in np.arange(0,len(res))]
	csi_param = [res[i][6] for i in np.arange(0,len(res))]
	pod_param = [res[i][7] for i in np.arange(0,len(res))]
	far_param = [res[i][8] for i in np.arange(0,len(res))]

	print(np.mean(csi_logit))
	print(np.mean(pod_logit))
	print(np.mean(far_logit))

	plt.figure();plt.boxplot([csi_logit, csi_cond, csi_param],labels=["logit","cond","t_totals"])
	plt.figure();plt.boxplot([pod_logit, pod_cond, pod_param],labels=["logit","cond","t_totals"])
	plt.figure();plt.boxplot([far_logit, far_cond, far_param],labels=["logit","cond","t_totals"])
	plt.show()

def rfe_selection(event, model, cv):

	#Use recursive feature elimination to find the N most important variables for logistic
	# regression

	#Load reanalysis data at stations, which has already been combined with event data
	if model == "era5":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"era5_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5")
	elif model == "barra":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra")
	else:
		raise ValueError("Invalid model name")
	
	#Get preds by taking the unique diagnostics which are in the top 20 ranked HSS for each type
	# of event (lightning, SCW given lightning, STA, AWS)
	preds = np.empty(0)
	for i in ["pss_light","pss_conv_aws","pss_sta","pss_conv_aws_cond_light"]:
		preds = np.append(preds,pss_df.sort_values(i, ascending=False).index[0:20].values)
	preds = np.unique(preds)

	#Clip the data at the 99.9 and 0.1 percentiles
	if event=="is_sta":
		upper = np.percentile(df_sta[preds],99.9, axis=0)
		lower = np.percentile(df_sta[preds],0.1, axis=0)
		for p in preds:
			df_sta.loc[df_sta.loc[:,p] >= (upper[preds==p])[0], p] = upper[preds==p]
			df_sta.loc[df_sta.loc[:,p] <= (lower[preds==p])[0], p] = lower[preds==p]
	elif event=="is_conv_aws":
		upper = np.percentile(df_aws[preds],99.9, axis=0)
		lower = np.percentile(df_aws[preds],0.1, axis=0)
		for p in preds:
			df_aws.loc[df_aws.loc[:,p] >= (upper[preds==p])[0], p] = upper[preds==p]
			df_aws.loc[df_aws.loc[:,p] <= (lower[preds==p])[0], p] = lower[preds==p]

	scaler = MinMaxScaler()
	logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
	
	if cv:
		rfecv = RFECV(estimator=logit, step=1, verbose=10,\
			scoring="roc_auc",n_jobs=-1,cv=5)
		if event=="is_sta":
			rfecv = rfecv.fit(scaler.fit_transform(df_sta[preds]), df_sta[event])
		elif event=="is_conv_aws":
			rfecv = rfecv.fit(scaler.fit_transform(df_aws[preds]), df_aws[event])
		pd.DataFrame({"preds":preds,"ranking":rfecv.ranking_, "support":rfecv.support_}).\
			to_csv("/g/data/eg3/ab4502/ExtremeWind/points/logit_preds_ranking_cv_"+\
			model+"_"+event+".pkl", index=False)
	else:
		rfe = RFE(estimator=logit, step=1, verbose=10,\
			n_features_to_select=1)
		if event=="is_sta":
			rfe = rfe.fit(scaler.fit_transform(df_sta[preds]), df_sta[event])
		elif event=="is_conv_aws":
			rfe = rfe.fit(scaler.fit_transform(df_aws[preds]), df_aws[event])
		pd.DataFrame({"preds":preds,"ranking":rfe.ranking_, "support":rfe.support_}).\
			to_csv("/g/data/eg3/ab4502/ExtremeWind/points/logit_preds_ranking_"+\
			model+"_"+event+".pkl", index=False)

def rfe_selection_custom(event, model, K=5):

	#Use recursive feature elimination to find the N most important variables for logistic
	# regression

	#Load diagnostics and events
	if model == "era5":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"era5_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5")
	elif model == "barra":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra")
	else:
		raise ValueError("Invalid model name")
	
	#Get preds by taking the unique diagnostics which are in the top 20 ranked HSS for each type
	# of event (lightning, SCW given lightning, STA, AWS)
	preds = np.empty(0)
	for i in ["pss_light","pss_conv_aws","pss_sta","pss_conv_aws_cond_light"]:
		preds = np.append(preds,pss_df.sort_values(i, ascending=False).index[0:20].values)
	preds = np.unique(preds)

	logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)

	pool = multiprocessing.Pool()
	from sklearn.model_selection import KFold
	train_dfs = []
	test_dfs = []
	if event=="is_sta":
		train_dfs = []
		test_dfs = []
		split = StratifiedShuffleSplit(n_splits=K,test_size=0.8,random_state=0)
		for test_index, train_index in split.split(X=df_sta, y=df_sta[event]):
			train_dfs.append(df_sta.iloc[train_index,:])
			test_dfs.append(df_sta.iloc[test_index,:])
	elif event=="is_conv_aws":
		train_dfs = []
		test_dfs = []
		split = StratifiedShuffleSplit(n_splits=K,test_size=0.8,random_state=0)
		for test_index, train_index in split.split(X=df_aws, y=df_aws[event]):
			train_dfs.append(df_aws.iloc[train_index,:])
			test_dfs.append(df_aws.iloc[test_index,:])
	
	#Clip the data at the 99.9 and 0.1 percentiles
	if event=="is_sta":
		upper = np.percentile(df_sta[preds],99.9, axis=0)
		lower = np.percentile(df_sta[preds],0.1, axis=0)
		for p in preds:
			df_sta.loc[df_sta.loc[:,p] >= (upper[preds==p])[0], p] = upper[preds==p]
			df_sta.loc[df_sta.loc[:,p] <= (lower[preds==p])[0], p] = lower[preds==p]
	elif event=="is_conv_aws":
		upper = np.percentile(df_aws[preds],99.9, axis=0)
		lower = np.percentile(df_aws[preds],0.1, axis=0)
		for p in preds:
			df_aws.loc[df_aws.loc[:,p] >= (upper[preds==p])[0], p] = upper[preds==p]
			df_aws.loc[df_aws.loc[:,p] <= (lower[preds==p])[0], p] = lower[preds==p]

	scaler = MinMaxScaler()
	scaler = RobustScaler()
	#DO THIS RECURSIVELY
	###
	old_hss = -1
	new_hss = 0

	hss_all = []
	eliminated_preds = []
	#while new_hss > old_hss:
	while len(preds) > 1:
		temp_hss = new_hss

		coefs_list = []
		hss_list_ext = []
		for i in np.arange(K):
			print(i)
			mod = logit.fit(scaler.fit_transform(train_dfs[i][preds]),train_dfs[i][event])
			probs = mod.predict_proba(scaler.fit_transform(test_dfs[i][preds]))[:,1]
			
			hss_list_int = []
			for t in np.linspace(0,1,100):
				hits = float(((test_dfs[i][event]==1) & (probs>t)).sum())
				misses = float(((test_dfs[i][event]==1) & (probs<=t)).sum())
				fa = float(((test_dfs[i][event]==0) & (probs>t)).sum())
				cn = float(((test_dfs[i][event]==0) & (probs<=t)).sum())
				if (hits / (hits + misses)) > 0.66:
					hss = ( 2*(hits*cn - misses*fa) ) / \
						( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * \
						(hits + cn) )
				else:
					hss=0
				hss_list_int.append(hss)
			coefs_list.append(np.squeeze(mod.coef_))
			hss_list_ext.append(np.max(hss_list_int))
		preds = preds[~(preds == preds[np.argmin( abs(np.stack(coefs_list).mean(axis=0)))])]
		new_hss = np.mean(hss_list_ext)
		#old_hss = temp_hss
		hss_all.append(new_hss)
		eliminated_preds.append(preds[(preds == preds[np.argmin( abs(np.stack(coefs_list).\
			mean(axis=0)))])])



	###


	pd.DataFrame({"preds":preds,"ranking":rfecv.ranking_, "support":rfecv.support_}).\
		to_csv("/g/data/eg3/ab4502/ExtremeWind/points/logit_preds_ranking_"+\
		model+"_"+event+".pkl", index=False)


if __name__ == "__main__":

	logit_test(["ml_cape","srhe_left","ml_el",\
			"Umean06","s03",\
			"lr36","mhgt","qmeansubcloud","qmean01","te_diff"],\
		"era5","is_conv_aws",["t_totals"])

