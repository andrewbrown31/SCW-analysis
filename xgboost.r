library(xgboost)

test = read.csv("/g/data/eg3/ab4502/ExtremeWind/points/ml/erai_test_2010-2015.csv")
train = read.csv("/g/data/eg3/ab4502/ExtremeWind/points/ml/erai_train_2010-2015.csv")

#param_list = c("ml_cape","ml_cin",
#"mu_cin","mu_cape","srh01","srh03","srh06","scp",
#		"stp","ship","mmp","relhum850.500","crt","lr1000","lcl",
#		"relhum1000.700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",
#		"cape.s06","cape.ssfc6","month")
param_list = c("s06","mu_cape","lr1000","month","relhum850.500")
form = paste(param_list,collapse="+")
form = paste("objective",form,sep="~")

test_features = as.matrix(test[param_list])
train_features = as.matrix(train[param_list])

mod = xgboost(data=train_features,label=train$objective,nrounds=50,objective="binary:logistic",missing=NaN,nfold=10,max.depth=4)

preds = predict(mod,test_features)

preds_b = (preds>0.2)*1
obs = test$objective

hits = sum((preds_b==1)&(obs==1))
misses = sum((preds_b==0)&(obs==1))
false_alarm = sum((preds_b==1)&(obs==0))
correct_negative = sum((preds_b==0)&(obs==0))

print(hits/(hits+misses))
print(false_alarm/(hits+false_alarm))

plot(preds,type="l")
lines(preds*obs,col="red")

xgb.importance(model=mod,feature_names=colnames(train_features))

#temp_mod = glm(formula=objective~mu_cape+relhum850.500+lr1000+cape.s06,family=binomial(link="logit"),data=train)
temp_mod = glm(formula=form,data=train,family="binomial")

#Make prediction on the test data (current year) and save to array
preds = predict(object=temp_mod,newdata=test,type="response")
