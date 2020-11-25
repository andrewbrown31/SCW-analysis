import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
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
			"era5_allvars_v2_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5_v2")
	elif model == "barra":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_2005_2018_2.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra_fc")
	else:
		raise ValueError("Invalid model name")

	#Set the correct dataframe based on event type 
	if event=="is_sta":
		df = df_sta
	elif event=="is_conv_aws":
		df = df_aws
	
	#Create 16 unique train/test datasets with balanced number of events
	pool = multiprocessing.Pool()
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

	pd.DataFrame({"predictors":param_out, "hss":hss_out, "pss":pss_out,\
		"pss_thresh":pss_thresh_out, "hss_thresh":hss_thresh_out}).\
		to_csv("/g/data/eg3/ab4502/ExtremeWind/points/logit_skill_"+model+"_v2_"+event+".csv",\
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


def run_logit():

	#BARRA
	logit = LogisticRegression(class_weight="balanced", solver="liblinear")
	pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"barra_allvars_2005_2018_2.pkl", T=1000, compute=False, l_thresh=2,\
		is_pss="hss", model_name="barra_fc")
	#Convective AWS
	preds = ["lr36","lr_freezing","ml_el","s06","srhe_left","Umean06"]
	event = "is_conv_aws"
	p = "t_totals"
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df_aws["logit"] = logit_mod.predict_proba(df_aws[preds])[:,1]
	res=[pss([t, df_aws, "logit", event, "hss"]) for t in np.linspace(0,1,100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_logit = hss_thresh[np.argmax(hss)]
	hss_logit = np.max(hss)
	res = [pss([t, df_aws, p, event, "hss"]) for t in \
		np.linspace(np.percentile(df_sta.loc[:,p],50),\
		    np.percentile(df_sta.loc[:,p],99.5),100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_p = hss_thresh[np.argmax(hss)]
	hss_p = np.max(hss)
	print("BARRA Conv AWS")
	print(p, "hss: ", hss_p, "thresh: ", hss_thresh_p)
	print("logit", "hss: ", hss_logit, "hss_thresh: ", hss_thresh_logit)
	#STA
	#preds = ["lr36","lr_freezing","mhgt","ml_el","s06","srhe_left","Umean06"]
	preds = ['qmean06', 'pwat', 'qmean01', 'sb_lcl', 'ddraft_temp']	
	event = "is_sta"
	p = "dcp"
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df_aws["logit"] = logit_mod.predict_proba(df_aws[preds])[:,1]
	res=[pss([t, df_aws, "logit", event, "hss"]) for t in np.linspace(0,1,100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_logit = hss_thresh[np.argmax(hss)]
	hss_logit = np.max(hss)
	res = [pss([t, df_aws, p, event, "hss"]) for t in \
		np.linspace(np.percentile(df_sta.loc[:,p],50),\
		    np.percentile(df_sta.loc[:,p],99.5),100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_p = hss_thresh[np.argmax(hss)]
	hss_p = np.max(hss)
	print("BARRA STA")
	print(p, "hss: ", hss_p, "thresh: ", hss_thresh_p)
	print("logit", "hss: ", hss_logit, "hss_thresh: ", hss_thresh_logit)

	#ERA5
	pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"era5_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
		is_pss="hss", model_name="era5")
	#Convective AWS
	preds = ["lr36","mhgt","ml_el","qmean01","srhe_left","Umean06"]
	event = "is_conv_aws"
	p = "t_totals"
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df_aws["logit"] = logit_mod.predict_proba(df_aws[preds])[:,1]
	res=[pss([t, df_aws, "logit", event, "hss"]) for t in np.linspace(0,1,100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_logit = hss_thresh[np.argmax(hss)]
	hss_logit = np.max(hss)
	res = [pss([t, df_aws, p, event, "hss"]) for t in \
		np.linspace(np.percentile(df_sta.loc[:,p],50),\
		    np.percentile(df_sta.loc[:,p],99.5),100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_p = hss_thresh[np.argmax(hss)]
	hss_p = np.max(hss)
	print("ERA5 Conv AWS")
	print(p, "hss: ", hss_p, "thresh: ", hss_thresh_p)
	print("logit", "hss: ", hss_logit, "hss_thresh: ", hss_thresh_logit)
	#STA
	preds = ["lr36","ml_cape","srhe_left","Umean06"]
	event = "is_sta"
	p = "dcp"
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df_aws["logit"] = logit_mod.predict_proba(df_aws[preds])[:,1]
	res=[pss([t, df_aws, "logit", event, "hss"]) for t in np.linspace(0,1,100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_logit = hss_thresh[np.argmax(hss)]
	hss_logit = np.max(hss)
	res = [pss([t, df_aws, p, event, "hss"]) for t in \
		np.linspace(np.percentile(df_sta.loc[:,p],50),\
		    np.percentile(df_sta.loc[:,p],99.5),100)]
	hss = [res[i][0] for i in np.arange(len(res))]
	hss_thresh = [res[i][1] for i in np.arange(len(res))]
	hss_thresh_p = hss_thresh[np.argmax(hss)]
	hss_p = np.max(hss)
	print("ERA5 STA")
	print(p, "hss: ", hss_p, "thresh: ", hss_thresh_p)
	print("logit", "hss: ", hss_logit, "hss_thresh: ", hss_thresh_logit)

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
			"barra_allvars_2005_2018_2.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra_fc")
	else:
		raise ValueError("Invalid model name")
	
	#Get preds by taking the unique diagnostics which are in the top 20 ranked HSS for each type
	# of event (lightning, SCW given lightning, STA, AWS)
	#preds = np.empty(0)
	#for i in ["pss_light","pss_conv_aws","pss_sta","pss_conv_aws_cond_light"]:
	#	preds = np.append(preds,pss_df.sort_values(i, ascending=False).index[0:20].values)
	#preds = np.unique(preds)
	preds = np.array(['ml_cape', 'mu_cape', 'sb_cape',\
	     'ml_cin', 'sb_cin', 'mu_cin', 'ml_lcl', 'mu_lcl', 'sb_lcl', 'eff_cape',\
	     'eff_cin', 'eff_lcl', 'lr01', 'lr03', 'lr13', 'lr36', 'lr24', 'lr_freezing',\
	     'lr_subcloud', 'qmean01', 'qmean03', 'qmean06', 'qmeansubcloud', 'q_melting',\
	     'q1', 'q3', 'q6', 'rhmin01', 'rhmin03', 'rhmin13', 'rhminsubcloud', 'tei', 'wbz',\
	     'mhgt', 'mu_el', 'ml_el', 'sb_el', 'eff_el', 'pwat', \
	     'te_diff', 'dpd850', 'dpd700', 'dcape', 'ddraft_temp', 'sfc_thetae',\
	     'srhe_left', 'srh01_left', 'srh03_left', 'srh06_left', 'ebwd', 's010', 's06',\
	     's03', 's01', 's13', 's36', 'scld', 'U500', 'U10', 'U1', 'U3', 'U6', 'Ust_left',\
	     'Usr01_left', 'Usr03_left', 'Usr06_left', 'Uwindinf', 'Umeanwindinf',\
	     'Umean800_600', 'Umean06', 'Umean01', 'Umean03', 'wg10' ])

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

def colin_test():

	#Test the collinearity of the logistic equations by using VFE
	from sklearn.metrics import r2_score
	from scipy.stats import spearmanr

	#BARRA
	logit = LogisticRegression(class_weight="balanced", solver="liblinear")
	pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"barra_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
		is_pss="hss", model_name="barra_fc_v5")
	#Convective AWS
	event = "is_conv_aws"
	preds = ["ebwd","lr13","ml_el","Umean03","rhmin03"]
	vifs = [vif(np.array(df_aws[preds]), i) for i in np.arange(len(preds))]
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df1 = pd.DataFrame({"VIF":vifs, "coefs":np.squeeze(logit_mod.coef_)}, index=preds)
	
	preds = ["ebwd","ml_el","Umean03","rhmin03"]
	vifs = [vif(np.array(df_aws[preds]), i) for i in np.arange(len(preds))]
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df2 = pd.DataFrame({"VIF":vifs, "coefs":np.squeeze(logit_mod.coef_)}, index=preds)
	
	(pd.concat([df1, df2], axis=1)).to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/vif_barra_aws.csv", float_format="%.2e")
	print(pd.concat([df1, df2], axis=1))

	#Test CV HSS scores
	#preds = ["eff_lcl","U1","sb_cape","lr13","rhmin03","eff_cin"]
	#barra_aws = logit_predictor_test("barra", "is_conv_aws", preds, "t_totals", 16)

	#STA
	preds = ["ebwd","Umean800_600","lr13","rhmin13","ml_el"]
	vifs = [vif(np.array(df_aws[preds]), i) for i in np.arange(len(preds))]
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df1 = pd.DataFrame({"VIF":vifs, "coefs":np.squeeze(logit_mod.coef_)}, index=preds)

	print(df1)

	#ERA5
	pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"era5_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
		is_pss="hss", model_name="era5_v5")
	#Convective AWS
	preds = ["ebwd","Umean800_600","lr13","rhmin13","srhe_left","q_melting","eff_lcl"]
	vifs = [vif(np.array(df_aws[preds]), i) for i in np.arange(len(preds))]
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df1 = pd.DataFrame({"VIF":vifs, "coefs":np.squeeze(logit_mod.coef_)}, index=preds)
	preds = ["ebwd","Umean800_600","rhmin13","srhe_left","q_melting","eff_lcl"]
	vifs = [vif(np.array(df_aws[preds]), i) for i in np.arange(len(preds))]
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df2 = pd.DataFrame({"VIF":vifs, "coefs":np.squeeze(logit_mod.coef_)}, index=preds)
	(pd.concat([df1, df2], axis=1)).to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/vif_era5_aws.csv", float_format="%.2e")
	print(pd.concat([df1, df2], axis=1))
	#Test CV HSS scores
	#preds = ["ml_el","Umean03","eff_lcl","dpd700","rhmin01"]
	#era5_aws = logit_predictor_test("era5", "is_sta", preds, "t_totals", 16)

	#STA
	preds = ["ml_cape","Umean06","ebwd","lr13"]
	vifs = [vif(np.array(df_aws[preds]), i) for i in np.arange(len(preds))]
	logit_mod = logit.fit(df_aws[preds], df_aws[event])
	df1 = pd.DataFrame({"VIF":vifs, "coefs":np.squeeze(logit_mod.coef_)}, index=preds)
	print(df1)

def logit_predictor_test(model, event, preds, param, n_splits):

	#Test the performance of a set of predictors (preds) relative to a set of params with given thresholds
	#Output the mean HSS (over 16 CVs), the absolute HSS (trained/tested on the same dataset) and the absolute AUC

	import warnings
	np.random.seed(seed=0)
	warnings.simplefilter("ignore")

	#Load diagnostics/events
	if model == "era5":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"era5_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5_v5")
	elif model == "barra":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra_fc_v5")
	else:
		raise ValueError("Invalid model name")

	#Set the correct dataframe based on event type 
	if event=="is_sta":
		df = df_sta
	elif event=="is_conv_aws":
		df = df_aws

	train_dfs = []
	test_dfs = []
	split = StratifiedShuffleSplit(n_splits=n_splits,test_size=0.8,random_state=0)
	for test_index, train_index in split.split(X=df, y=df[event]):
		train_dfs.append(df.iloc[train_index,:])
		test_dfs.append(df.iloc[test_index,:])
    
	res = []
	thresh = []
	param_thresh = []
	param_res = []
	test_thresh = np.linspace(df.loc[:,param].min(), np.percentile(df.loc[:,param],99.95), 1000)
	pool = multiprocessing.Pool()
	for i in tqdm.tqdm(np.arange(len(train_dfs))):
		temp_hss,_,temp_thresh,_ = logit_train([i, train_dfs, test_dfs, preds, event])
		res.append(temp_hss)
		thresh.append(temp_thresh)

		iterable = itertools.product(test_thresh, [test_dfs[i]], [param], [event], ["hss"])
		res2 = pool.map(pss, iterable)
		temp_param_thresh = [res2[i][1] for i in np.arange(len(res2))]
		pss_p = [res2[i][0] for i in np.arange(len(res2))]
		param_res.append(np.max(pss_p))
		param_thresh.append(temp_param_thresh[np.argmax(pss_p)])
			
	hss_cv = np.mean([res for i in np.arange(len(test_dfs))])
	hss_min = np.min([res for i in np.arange(len(test_dfs))])
	hss_max = np.max([res for i in np.arange(len(test_dfs))])
	avg_thresh = np.mean(thresh)
	hss_param = np.mean([param_res for i in np.arange(len(test_dfs))])
	hss_param_max = np.max([param_res for i in np.arange(len(test_dfs))])
	hss_param_min = np.min([param_res for i in np.arange(len(test_dfs))])
	hss,_,opt_thresh,_ = logit_train([0, [df], [df], preds, event])
	print("logit_cv: ", hss_cv, param+" cv: ", np.mean(param_res), "logit total: ", hss)
	return hss, hss_cv, hss_min, hss_max, avg_thresh, opt_thresh, hss_param, param, hss_param_min, hss_param_max, np.mean(param_thresh)

def fwd_selection(model, event, pval_choice):

	#Perform the following procedure:
	#   1) Load the model data for the model and event given
	#   2) Create an intercept model using statsmodels
	#   3) For each variable, add the variable to the model and assess the p-value
	#   4) Accept a model for the next round either by using the lowest p-value (pval_choice==True), or by using HSS (pval_choice==False)

	print("INFO: Forward selection of variables for "+model+" using "+event)

	#Load diagnostics/events
	if model == "era5":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"era5_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5_v5")
	elif model == "barra":
		pss_df, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra_fc_v5")
	else:
		raise ValueError("Invalid model name")

	#Set the correct dataframe based on event type 
	if event=="is_sta":
		df = df_sta.reset_index().drop(columns="index")
	elif event=="is_conv_aws":
		df = df_aws.reset_index().drop(columns="index")

	#Test predictors are all "variables" available
	preds = np.array(['ml_cape', 'mu_cape', 'sb_cape',\
	     'ml_cin', 'sb_cin', 'mu_cin', 'ml_lcl', 'mu_lcl', 'sb_lcl', 'eff_cape',\
	     'eff_cin', 'eff_lcl', 'lr01', 'lr03', 'lr13', 'lr36', 'lr24', 'lr_freezing',\
	     'lr_subcloud', 'qmean01', 'qmean03', 'qmean06', 'qmeansubcloud', 'q_melting',\
	     'q1', 'q3', 'q6', 'rhmin01', 'rhmin03', 'rhmin13', 'rhminsubcloud', 'tei', 'wbz',\
	     'mhgt', 'mu_el', 'ml_el', 'sb_el', 'eff_el', 'pwat', \
	     'te_diff', 'dpd850', 'dpd700', 'dcape', 'ddraft_temp', 'sfc_thetae',\
	     'srhe_left', 'srh01_left', 'srh03_left', 'srh06_left', 'ebwd', 's010', 's06',\
	     's03', 's01', 's13', 's36', 'scld', 'U500', 'U10', 'U1', 'U3', 'U6', 'Ust_left',\
	     'Usr01_left', 'Usr03_left', 'Usr06_left', 'Uwindinf', 'Umeanwindinf',\
	     'Umean800_600', 'Umean06', 'Umean01', 'Umean03'])

	#Initialise things
	from plot_param import resample_events
	from statsmodels.tools.tools import add_constant
	from statsmodels.discrete.discrete_model import Logit 
	import warnings
	warnings.simplefilter("ignore")
	logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
	pool = multiprocessing.Pool()
	N=1000
	np.random.seed(seed=0)

	#Train model with statsmodel
	mod = Logit(df[event],add_constant(df[preds])["const"]).fit()

	#Train model with sklearn and get HSS
	logit_mod = logit.fit(add_constant(df[preds])[["const"]], df[event])
	df["predict"] = logit_mod.predict_proba(add_constant(df[preds])[["const"]])[:,1]
	iterable = itertools.product(np.linspace(0,1,100), [df[["predict", event]]], ["predict"], [event], ["hss"])
	res2 = pool.map(pss, iterable)
	current_hss = np.max([res2[i][0] for i in np.arange(len(res2))])

	statsmod_preds = []
	statsmod_hss = []
	alph = 0.05
	is_pval = True		#Keep track of overall progress (i.e. whether or not to continue)
	while is_pval:
		pval_ls = []		#Keep track of the p-value of each individual added param
		is_pval_ls = []		#Keep track of if all coefficients within the added-parameter model are significant
		hss_ls = []		#Keep track of the HSS
		hss_thresh = []		#Keep track of the HSS thresh
		for p in tqdm.tqdm(preds):
			if p not in statsmod_preds:
				mod = Logit(df[event],add_constant(df[statsmod_preds + [p]])).fit(disp=False)
				param_pval = mod.summary2().tables[1].loc[p, "P>|z|"]
				pval = mod.summary2().tables[1].loc[:, "P>|z|"]
				pval_ls.append(param_pval)
				is_pval_ls.append(all(pval <= alph))
				if not pval_choice:
					logit_mod = logit.fit(df[statsmod_preds + [p]], df[event])
					df["predict"] = logit_mod.predict_proba(df[statsmod_preds + [p]])[:,1]
					iterable = itertools.product(np.linspace(0,1,100), [df[["predict", event]]], ["predict"], [event], ["hss"])
					res2 = pool.map(pss, iterable)
					hss_ls.append(np.max([res2[i][0] for i in np.arange(len(res2))]))
			else:
				pval_ls.append(1)
				is_pval_ls.append(False)
				hss_ls.append(0)
		#If using pvalues to decide which variable to add, then chose the one with the minimum pvalue
		if pval_choice:
			if (min(pval_ls) <= alph) & (is_pval_ls[np.argmin(pval_ls)]):
				is_pval = True
				statsmod_preds.append(preds[np.argmin(pval_ls)])
				print("INFO: There are "+str(np.sum(is_pval_ls))+" new models which add value based on p-value")
				print("INFO: The min p-value is "+str(np.min(pval_ls))+" based on "+preds[np.argmin(pval_ls)])
			else:
				print("INFO: Stopping at "+str(len(statsmod_preds))+" variables")
				is_pval = False
		#Else, use the optimised HSS to decide (note that a different module is used to fit the model)
		else:
			if any((hss_ls > current_hss) & (is_pval_ls)):	    #If there is at least one predictor with a higher HSS and significant coef.
				for z in np.arange(len(is_pval_ls)):	    #Remove variables which add HSS but don't have a significant coef.
					if not is_pval_ls[z]:
						hss_ls[z] = 0
				#Calculate bootstrapped HSS, and get the upper 5%
				if len(statsmod_preds) >= 1:
					print("Bootstrapping the HSS to get confidence...")
					logit_mod = logit.fit(df[statsmod_preds], df[event])
					df["predict"] = logit_mod.predict_proba(df[statsmod_preds])[:,1]
					iterable = itertools.product(np.linspace(0,1,100), [df[["predict", event]]], ["predict"], [event], ["hss"])
					res2 = pool.map(pss, iterable)
					hss_temp = [res2[i][0] for i in np.arange(len(res2))]
					hss_thresh = [res2[i][1] for i in np.arange(len(res2))][np.argmax(hss_temp)]
					hss_boot = []
					event_ind, non_inds = resample_events(df, event, N, df[event].sum())
					for i in tqdm.tqdm(np.arange(N)):
						iterable = itertools.product([hss_thresh],\
							[df.iloc[np.append(event_ind[i], non_inds[i])][["predict", event]]],\
							["predict"], [event], ["hss"])
						res2 = pool.map(pss, iterable)
						hss_boot.append(res2[0][0])
				else:
					hss_boot = [0]
				#If the hss of the most skillful predictor is greater than the 95th percentile, then select the predictor and keep going.
				#Else, halt the proceudre
				if np.max(hss_ls) >= np.percentile(hss_boot, 95):   
					is_pval = True
					statsmod_preds.append(preds[np.argmax(hss_ls)])
					statsmod_hss.append(np.max(hss_ls))
					print("INFO: There are "+str(np.sum((hss_ls > np.percentile(hss_boot, 95)) &\
						(is_pval_ls)))+" new models which add value based on HSS and p-values")
					print("INFO: The max HSS is "+str(np.max(hss_ls))+" based on "+preds[np.argmax(hss_ls)])
					current_hss = max(hss_ls)
				else:
					is_pval = False
					print("INFO: Stopping at "+str(len(statsmod_preds))+" variables")
			else:
				is_pval = False
				print("INFO: Stopping at "+str(len(statsmod_preds))+" variables")

	#Now save the output
	logit_mod = logit.fit(df[statsmod_preds], df[event])
	mod = Logit(df[event],add_constant(df[statsmod_preds])).fit(disp=False)
	pval = mod.summary2().tables[1].loc[:, "P>|z|"]
	out_df = pd.DataFrame({"coef":np.squeeze(logit_mod.coef_), "non_cv_hss":statsmod_hss}, index=statsmod_preds)
	out_df.loc["const", "coef"] = logit_mod.intercept_
	out_df.loc["const", "non_cv_hss"] = 0
	out_df.loc[:, "p-val"] = pval
	if pval_choice:
		pval_str = "pval"
	else:
		pval_str = "hss" 
	out_df.to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/logit_fwd_sel_"+model+"_"+event+"_"+pval_str+"_v3.csv")

def logit_explicit_cv(outname):

	#Test logit models with explicitly defined predictors
    
	preds = ["ebwd","lr13","ml_el","Umean03","rhmin03"]
	barra_aws = logit_predictor_test("barra", "is_conv_aws", preds, "t_totals", 16)
	preds = ["ebwd","Umean800_600","lr13","rhmin13","ml_el"]
	barra_sta = logit_predictor_test("barra", "is_sta", preds, "mlcape*s06", 16)
	preds = ["ebwd","Umean800_600","lr13","rhmin13","srhe_left","q_melting","eff_lcl"]
	era5_aws = logit_predictor_test("era5", "is_conv_aws", preds, "t_totals", 16)
	preds = ["ml_cape","Umean06","ebwd","lr13"]
	era5_sta = logit_predictor_test("era5", "is_sta", preds, "dcp", 16)

	#Name each of the models and put into a list
	ind = ["barra_aws","barra_sta","era5_aws","era5_sta"]
	out_list = [barra_aws, barra_sta, era5_aws, era5_sta]

	hss = []; hss_cv = []; hss_min = []; hss_max = []; hss_param = []; param = []; avg_thresh = []; opt_thresh = []; hss_param_min = []; hss_param_max = []; param_thresh_cv = []
	for out in out_list:
		hss.append(out[0])
		hss_cv.append(out[1])
		hss_min.append(out[2])
		hss_max.append(out[3])
		avg_thresh.append(out[4])
		opt_thresh.append(out[5])
		hss_param.append(out[6])
		param.append(out[7])
		hss_param_min.append(out[8])
		hss_param_max.append(out[9])
		param_thresh_cv.append(out[10])
	pd.DataFrame({"hss":hss, "hss_cv":hss_cv, "hss_min":hss_min, "hss_max":hss_max, "thresh_cv":avg_thresh, "thresh":opt_thresh,\
	    "param":param, "hss_param":hss_param, "hss_param_min":hss_param_min, "hss_param_max":hss_param_max, "param_thresh_cv":param_thresh_cv},\
	    index=ind).to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/"+outname)

def plot_roc():

	_, era5_aws, era5_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"era5_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="era5_v5")
	_, barra_aws, barra_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"barra_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
			is_pss="hss", model_name="barra_fc_v5")

	barra_aws_preds = ["ebwd","lr13","ml_el","Umean03","rhmin03"]
	barra_sta_preds = ["ebwd","Umean800_600","lr13","rhmin13","ml_el"]
	era5_aws_preds = ["ebwd","Umean800_600","lr13","rhmin13","srhe_left","q_melting","eff_lcl"]
	era5_sta_preds = ["ml_cape","Umean06","ebwd","lr13"]
	logit = LogisticRegression(class_weight="balanced", solver="liblinear",\
		max_iter=1000)
	era5_aws.loc[:,"logit"] = logit.fit(era5_aws[era5_aws_preds], era5_aws["is_conv_aws"]).predict_proba(era5_aws[era5_aws_preds])[:,1]
	era5_sta.loc[:,"logit"] = logit.fit(era5_sta[era5_sta_preds], era5_sta["is_sta"]).predict_proba(era5_sta[era5_sta_preds])[:,1]
	barra_aws.loc[:,"logit"] = logit.fit(barra_aws[barra_aws_preds], barra_aws["is_conv_aws"]).predict_proba(barra_aws[barra_aws_preds])[:,1]
	barra_sta.loc[:,"logit"] = logit.fit(barra_sta[barra_sta_preds], barra_sta["is_sta"]).predict_proba(barra_sta[barra_sta_preds])[:,1]

	plt.close(); plt.figure(figsize=[10,8]); 
	matplotlib.rcParams.update({'font.size': 12})
	plot_roc_fn(barra_aws, "t_totals", "is_conv_aws", -22000, 2000, -200, 60, 2, 2, 1, "BARRA Measured", "T-Totals")
	plot_roc_fn(barra_sta, "mlcape*s06", "is_sta", -6000, 1000, -200, 200, 2, 2, 2, "BARRA Reported", "MLCS6")
	plot_roc_fn(era5_aws, "t_totals", "is_conv_aws", -20000, 2000, -200, 60, 2, 2, 3, "ERA5 Measured", "T-Totals")
	plot_roc_fn(era5_sta, "dcp", "is_sta", -1000, 750, -200, 200, 2, 2, 4, "ERA5 Reported", "DCP")
	plt.savefig("figA5.eps", bbox_inches="tight", dpi=300)

def plot_roc_fn(df, p, event, end_p, step_p, end_logit, step_logit, rows, cols, subplot_no, title, pname):

	plt.subplot(2,2,subplot_no)
	if subplot_no in [3,4]:
		plt.xlabel('False Positive Rate')
	if subplot_no in [1,3]:
		plt.ylabel('True Positive Rate')
	lw=2
	fpr_logit, tpr_logit, thresh_logit = roc_curve(df[event], df["logit"])
	roc_auc_logit = auc(fpr_logit, tpr_logit)
	plt.plot(fpr_logit, tpr_logit, color='tab:blue',
		 lw=lw, label='ROC curve Logit reg. (area = %0.2f)' % roc_auc_logit)
	fpr_p, tpr_p, thresh_p = roc_curve(df[event], df[p])
	roc_auc_p = auc(fpr_p, tpr_p)
	plt.plot(fpr_p, tpr_p, color='darkorange',
		 lw=lw, label='ROC curve '+pname+' (area = %0.2f)' % roc_auc_p)
	plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.title(title)
	plt.legend(loc="lower right", fontsize="x-small")
	for i, txt in enumerate(thresh_logit[200:end_logit:step_logit]):
	    plt.annotate(txt.round(2), (fpr_logit[200:end_logit:step_logit][i], tpr_logit[200:end_logit:step_logit][i]), fontsize=12, weight="bold")
	plt.plot(fpr_logit[200:end_logit:step_logit], tpr_logit[200:end_logit:step_logit], color="tab:blue", linestyle="none", marker="o")
	for i, txt in enumerate(thresh_p[0:end_p:step_p]):
		if p == "t_totals":
			plt.annotate(txt.round(1), (fpr_p[0:end_p:step_p][i]+0.05, tpr_p[0:end_p:step_p][i]), fontsize=12, weight="bold")
		elif p == "dcp":
			plt.annotate(txt.round(3), (fpr_p[0:end_p:step_p][i], tpr_p[0:end_p:step_p][i]), fontsize=12, weight="bold")
		else:
			plt.annotate(str(int(txt)), (fpr_p[0:end_p:step_p][i], tpr_p[0:end_p:step_p][i]-0.08), fontsize=12, weight="bold")
	plt.plot(fpr_p[0:end_p:step_p], tpr_p[0:end_p:step_p], color="tab:orange", linestyle="none", marker="o")
	plt.axhline(0.667, color="k", linestyle="--")

if __name__ == "__main__":

	#run_logit()
	#TODO: As well as choosing models by the HSS, use bootstrapping to decide if it is a statistically significant increase?
	#fwd_selection("era5", "is_sta", False)
	#fwd_selection("barra", "is_sta", False)
	#fwd_selection("barra", "is_conv_aws", False)
	#fwd_selection("era5", "is_conv_aws", False)

	#logit_explicit_cv("logit_cv_skill_v3.csv")
	plot_roc()
	#colin_test()
