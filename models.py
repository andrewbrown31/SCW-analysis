#Try and fit a variety of models to convective wind gust data

from event_analysis import *
from plot_param import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def load_data(season=False):
	#Load ERA-Interim data 2010-2015
	print("Loading ERA-Interim variables...");erai_df_fc = load_erai_df(False,True)
	erai_df_fc = erai_df_fc.drop_duplicates(subset=["lat","lon","year","month","day","hour"])
	erai_df_an = load_erai_df(False,False)
	erai_df_an = erai_df_an.set_index(["date","loc_id"])
	erai_df_fc = erai_df_fc.set_index(["date","loc_id"])
	erai_df = pd.concat([erai_df_fc.wg10,erai_df_an],axis=1)

	#Load wind gust data and lightning data, combine with ERA-Interim into one dataframe
	print("Loading AWS...");aws = load_aws_all(resample=True)
	lightning=load_lightning()
	aws = aws.set_index(["stn_name"],append=True)
	lightning = lightning.set_index(["date","loc_id"])
	erai_df = pd.concat([aws.wind_gust,erai_df,lightning.lightning],axis=1)
	erai_df = erai_df[~(erai_df.lat.isna()) & ~(erai_df.wind_gust.isna())]

	#If specified, extract only the "warm" or "cold" months from the data
	if season != False:
		warm_inds = np.array([erai_df.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,erai_df.shape[0])])
		if season == "warm":
			erai_df = erai_df[warm_inds]
		elif season == "cold":
			erai_df = erai_df[~warm_inds]

	#Create wind gust events we are interested in
	erai_df["events"] = ((erai_df.lightning>=2) & (erai_df.wind_gust>=20))*1

	#Return dataframe with important variables
	param_list = ["mu_cape","relhum850-500","lr1000","lcl","s06","wg10","srh03","ssfc1",\
		"ssfc3","mu_cin","mmp","td800","cape*s06","events"]
	erai_df = erai_df[np.array(param_list)].dropna()
	
	return erai_df

def load_jdh_events():

	df,jdh_df,non_jdh_df = analyse_jdh_events()
	df_warm = df[np.in1d(df.month,np.array([10,11,12,1,2,3]))]
	df_cool = df[np.in1d(df.month,np.array([4,5,6,7,8,9]))]

	return (df,df_warm,df_cool)

def random_forest(train,test,features):
	rfc = RandomForestClassifier()
	rf_model = rfc.fit(train[features],train["events"])
	probs = rf_model.predict_proba(test[features])
	obs = np.array(test["events"])
	preds = rf_model.predict(test[features])

	return (probs,preds,obs)	

def decision_tree(train,test,features):
	tree = DecisionTreeClassifier()
	tree_model = tree.fit(train[features],train["events"])
	probs = tree_model.predict_proba(test[features])
	obs = np.array(test["events"])
	preds = tree_model.predict(test[features])
	
	return (probs,preds,obs)	

def logit(train,test,features):
	logit = LogisticRegression()
	logit_model = logit.fit(train[features],train["events"])
	probs = logit_model.predict_proba(test[features])
	obs = np.array(test["events"])
	preds = logit_model.predict(test[features])
	
	return (probs,preds,obs)	

def print_stats(preds,obs,model):
	hits = ((preds==1) & (obs==1)).sum().astype(float)
	misses = ((preds==0) & (obs==1)).sum().astype(float)
	false_alarm = ((preds==1) & (obs==0)).sum().astype(float)
	correct_negative = ((preds==0) & (obs==0)).sum().astype(float)
	print("		MODEL = "+model+\
		"\nHITS = "+str(hits)+"\nMISSES = "+str(misses)+"\nFALSE ALARMS = "+\
		str(false_alarm)+"\nCORRECT NEGATIVE = "+str(correct_negative)+"\n")

if __name__=="__main__":
	#erai_df = load_data(season=False)	#season can be False, warm or cold
	df,warm_df,cool_df = load_jdh_events()
	train,test = train_test_split(df.rename(columns={"jdh":"events"}))
	feats = ["mu_cape","mu_cin","relhum850-500",\
			"s06","wg10"]
	probs,preds,obs = random_forest(train,test,feats)
	print_stats(preds,obs,"RF")
	probs,preds,obs = logit(train,test,feats)
	print_stats(preds,obs,"LOGISTIC REGRESSION")
	probs,preds,obs = decision_tree(train,test,feats)
	print_stats(preds,obs,"DECISION TREE")


