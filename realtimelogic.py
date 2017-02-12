import pandas
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
heart = pandas.read_csv("pc.csv")
heart.loc[heart["heartpred"]==2,"heartpred"]=1
heart.loc[heart["heartpred"]==3,"heartpred"]=1
heart.loc[heart["heartpred"]==4,"heartpred"]=1
heart["slope"] = heart["slope"].fillna(heart["slope"].median())
heart["thal"] = heart["thal"].fillna(heart["thal"].median())
heart["ca"] = heart["ca"].fillna(heart["ca"].median())
print(heart.describe())
predictors=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
alg=LogisticRegression(random_state=1)
predictions = []
train_predictors = (heart[predictors].iloc[:,:])
train_target = heart["heartpred"].iloc[:]
alg.fit(train_predictors, train_target)
l=len(heart.index)
while True:
	x1=raw_input("Input age (67.0):")
	x2=raw_input("Input sex (1.0):")
	x3=raw_input("Input cp (4.0):")
	x4=raw_input("Input trestbps (160.0):")
	x5=raw_input("Input chol (286.0):")
	x6=raw_input("Input fbs (0.0):")
	x7=raw_input("Input restecg (2.0):")
	x8=raw_input("Input thalach (108.0):")
	x9=raw_input("Input exang (1.0):")
	x10=raw_input("Input oldpeak (1.5):")
	x11=raw_input("Input slope (2.0):")
	x12=raw_input("Input ca (3.0):")
	x13=raw_input("Input thal (3.0):")
	f = open('pc.csv','a+')
	f.write(x1+','+x2+','+x3+','+x4+','+x5+','+x6+','+x7+','+x8+','+x9+','+x10+','+x11+','+x12+','+x13+',')
	f.close()
	heart = pandas.read_csv("pc.csv")
	#print(l+1)
	print(heart[predictors].iloc[l,:])
	test_predictions = alg.predict(heart[predictors].iloc[l,:])
	test_predictions[test_predictions > .5] = 1
	test_predictions[test_predictions <=.5] = 0
	print(test_predictions.item(0))
	x14=test_predictions.item(0)
	print(type(x14))
	f = open('pc.csv','a+')
	f.write("%s"%(x14))
	f.write(os.linesep)
	f.close()
	l+=1
