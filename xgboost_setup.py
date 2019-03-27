from event_analysis import *
import xgboost as xgb

aws = load_aws_all(True)
erai_df = load_erai_df()

lightning=load_lightning()
aws = aws.set_index(["stn_name"],append=True)
erai_df = erai_df.set_index(["date","loc_id"])
lightning = lightning.set_index(["date","loc_id"])
erai_df = pd.concat([aws.wind_gust,erai_df,lightning.lightning],axis=1)
erai_df = erai_df[~(erai_df.lat.isna()) & ~(erai_df.wind_gust.isna())]

erai_df["objective"] = ((erai_df.lightning >= 2) & (erai_df.mu_cin <= 100) \
				& (erai_df.wind_gust >= 15))*1

test_erai = erai_df[0:2500]
train_erai = erai_df[2500:]

test_erai.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/ml/erai_test_2010-2015.csv")
train_erai.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/ml/erai_train_2010-2015.csv")

