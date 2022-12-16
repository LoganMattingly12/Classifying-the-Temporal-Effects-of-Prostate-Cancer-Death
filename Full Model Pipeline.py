# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:36:12 2022

@author: ltmat
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample as rs
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

#Detrmines the prognosis sample
survival_months = 120
#Sets the data normilization method
data_norm = "None"

#Initial Data Selection
scaler = StandardScaler()
df = pd.read_csv('C:\\Users\\ltmat\\Documents\\Logan\\Data Science\\Cancer\\Updated Recoded Prostate Data.csv',low_memory=False)
df.drop('Year of diagnosis',inplace=True,axis=1)
df['Survival months']=pd.cut(df['Survival months'], bins=[-1,survival_months,1000],labels=[0,1])
scaler = StandardScaler()
x = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
y = df.iloc[:,30]
x = pd.DataFrame(x)
y = pd.DataFrame(y)

train_data = pd.concat([x, y], axis = 1)
group_0 = train_data[train_data['Survival months']==0]
group_1 = train_data[train_data['Survival months']==1]
data_upsampled_0=group_0
data_upsampled_1 = rs(group_1,replace = True, n_samples = 90000,random_state=40)
upsampled =pd.concat([data_upsampled_0,data_upsampled_1])
dataframe = upsampled.dropna()
dataset = dataframe
x= dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print(y.value_counts())

# 5 fold-stratified cross validation: 
sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in sss.split(x,y):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
scaler.fit(X_train.fillna(0))


#LASSO Feature Selection
sel_ = SelectFromModel(LogisticRegression(C=0.0015, penalty="l1", solver='liblinear',random_state=7))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train.values.ravel())
sel_.get_support()
selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))
np.sum(sel_.estimator_.coef_ == 0)
removed_feats = X_train.columns[(sel_.estimator_.coef_ != 0).ravel().tolist()]
X_train = sel_.transform(X_train.fillna(0))
X_test= sel_.transform(X_test.fillna(0))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)

if data_norm == "SMOTE":
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    group_0 = train_data[train_data['Survival months']==0]
    group_1 = train_data[train_data['Survival months']==1]
    data_upsampled_0=group_0
    data_upsampled_1 = rs(group_1,replace = True, n_samples = 90000,random_state=40)
    upsampled =pd.concat([data_upsampled_0,data_upsampled_1])
    dataframe = upsampled.dropna()
    dataset = dataframe
    X_train = dataset.iloc[:, :-1]
    y_train = dataset.iloc[:, -1]
    print(y_train.value_counts())

if data_norm == 'RUS':
    train_data = pd.concat([X_train, y_train], axis = 1)
    group_0 = train_data[train_data['Survival months']==0]
    group_1 = train_data[train_data['Survival months']==1]
    data_upsampled_1=group_1
    data_upsampled_0 = rs(group_0,replace = True, n_samples =140000,random_state=40)
    upsampled =pd.concat([data_upsampled_0,data_upsampled_1])
    dataframe = upsampled.dropna()
    dataset = dataframe
    X_train = dataset.iloc[:, :-1]
    y_train = dataset.iloc[:, -1]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    y_test.value_counts()
    
else:
    pass
#train_data = pd.concat([X_train, y_train], axis = 1)

print(y_train.value_counts())

print('Data Compiled')


#Model Training
tf.random.set_seed(568)
y_train = np.ravel(y_train)
logr = linear_model.LogisticRegression(random_state=1234, max_iter=1000).fit(X_train,y_train) 
print('Logistic Regression Fit')


model = tf.keras.Sequential([
    tf.keras.layers.Dense((len(selected_feat)+1), activation='relu'),
    tf.keras.layers.Dense(np.mean(len(selected_feat)+1), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.0,nesterov=True,name="SGD"),
    metrics=[
     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
     tf.keras.metrics.Precision(name='precision'),
     tf.keras.metrics.Recall(name='recall')])
tensor = model.fit(X_train, y_train, epochs=10)
print('ANN Fit')

#LR Testing
y_pred_logr = logr.predict(X_test)

#Logistic Regression Metrics
print('Logistic Regression')
test2 = (accuracy_score(y_test,y_pred_logr)*100)
y_pred_proba = logr.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print('The accuracy of Testing is '+str(test2))
print('\n')
print(classification_report(y_test, y_pred_logr))
print('The AUC is '+str(auc))
cm = confusion_matrix(y_test,y_pred_logr,labels = [0,1])
print(cm)


#TF Testing
predictions = model.predict(X_test)
tf_ypred = [1 if prob >=
            0.5 else 0 for prob in np.ravel(predictions)]

#TF Model Metrics
print('\nTensorFlow')
test6 = (accuracy_score(y_test,tf_ypred)*100)
auc = metrics.roc_auc_score(y_test, tf_ypred)
print('\nThe accuracy of Testing is '+str(test6))
print('\n')
print(classification_report(y_test, tf_ypred,zero_division='warn'))
print('The AUC is '+str(auc))
cm3 = confusion_matrix(y_test,tf_ypred,labels = [0,1])
print(cm3)

#ROC Curves
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
fpr1,tpr1, _ = metrics.roc_curve(y_test, tf_ypred)
plt.plot(fpr,tpr)
plt.plot(fpr1,tpr1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()