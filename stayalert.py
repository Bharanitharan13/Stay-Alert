# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:25:05 2020

@author: Bharanitharan
"""
    
import pandas as pd
import numpy as np
import os 

### setting working directory
os.chdir('E:\\python\\stay alert')

train = pd.read_csv('fordTrain.csv')
test = pd.read_csv('fordTest.csv')


train.shape

test.shape

train.columns,test.columns
#### audit

train.head()

class_type =  train.dtypes

class_type
#### summary####
summary = train.describe()
summary

train['P8']

skewness_1 = train.skew()


#### drop of variable

train_1 = train.drop(['P8','V7','V9'],axis = 1)
test = test.drop(['P8','V7','V9'],axis = 1)

### checking missing values

train_1.isna().sum()

#### coorlizatio Matrix

cormat = train_1.corr()

cormat.to_csv('corr.csv')

train_1.columns
test.columns
TEST= test['IsAlert']



TEST

Y= train_1['IsAlert']
TEST_X = test.iloc[:,3:30]

X = train_1.iloc[:,3:30]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=123)

#xgboost

import xgboost as xgb

xgb_class = xgb.XGBClassifier(n_estimators = 20)

xgb_class.fit(x_train,y_train)

preds_class = xgb_class.predict(x_train)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(preds_class, y_train)
acc_score = accuracy_score(preds_class, y_train)
print(cm)
print(acc_score)

predic = xgb_class.predict(x_test)

cm = confusion_matrix(predic, y_test)
acc_score = accuracy_score(predic, y_test)
print(cm)
print(acc_score)

predic_test = xgb_class.predict(X_test)

cm = confusion_matrix(predic_test,Y_TEST)
acc_score = accuracy_score(predic_test, Y_TEST)
print(cm)
print(acc_score)


####running liner regression####
from sklearn.linear_model import LogisticRegression 

lr = LogisticRegression() 

lr.fit(x_train,y_train)

preds_lr = lr.predict(x_train)

lr.coef_
lr.intercept_
#####confution matix
from sklearn.metrics import confusion_matrix

Cm_lr = confusion_matrix(y_train,preds_lr)

mm =  pd.DataFrame(Cm_lr)

preds_lr_xtest = lr.predict(x_test)

### conufution matix fot xtest
Cm_lr_xtest = confusion_matrix(y_test,preds_lr_xtest)

mm_xtest = pd.DataFrame(Cm_lr_xtest)

mm_xtest.iloc[0][0] + mm_xtest.iloc[1][0] 
mm_xtest.iloc[1][0] 
mm_xtest

#### logistic using liner reg with normalize
from sklearn.preprocessing import normalize

x_train_scale = normalize(x_train)

lr.fit(x_train_scale,y_train)

preds_lr_normal = lr.predict(x_train_scale)

cm_lr_normal_xtrain = confusion_matrix(y_train,preds_lr_normal)

pd.DataFrame(cm_lr_normal_xtrain)


x_train_scale = pd.DataFrame(x_train_scale)


skewness = x_train_scale.skew()

skewness

### decison tree ######
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(x_train,y_train)

pred_dtc = DTC.predict(x_train)

cm_DTC = confusion_matrix(y_train,pred_dtc)

cm_DTC = pd.DataFrame(cm_DTC)

cm_DTC

pred_dtc_test = DTC.predict(x_test)

cm_dtc_test = confusion_matrix(y_test,pred_dtc_test)

pd.DataFrame(cm_dtc_test)

totalerr = (cm_dtc_test[0][1] + cm_dtc_test[1][0])

total = sum(cm_dtc_test)
pd.DataFrame(total)
total = sum(total)
missclass = (totalerr / total)*100
missclass


#### validing in TEST data set

test_x = test.drop(['P8','V7','V9'],axis = 1)

test_x = test_x.iloc[:,3:]

pred_test_x = DTC.predict(test_x)

Solution = pd.read_csv('Solution.csv')

cm_final = confusion_matrix(Solution['Prediction'],pred_test_x)

pd.DataFrame(cm_final)

#importing library
from sklearn.neural_network import MLPClassifier

# sk learns nural network

MLP = MLPClassifier(hidden_layer_sizes=(10,10,10), verbose = True, solver='sgd', max_iter=300)
MLP.fit(x_train,y_train)


from sklearn.metrics import accuracy_score

pred_mlp = MLP.predict(x_train)

acq_score_nural = accuracy_score(y_train,pred_mlp)
acq_score_nural

pred_ytest_MLP = MLP.predict(x_test)
acq_for_TEST = accuracy_score(pred_ytest_MLP,y_test)
acq_for_TEST

# in validation data set
pred_val = MLP.predict(test_x)
acq_for_val = accuracy_score(pred_val,Solution['Prediction'])

acq_for_val






