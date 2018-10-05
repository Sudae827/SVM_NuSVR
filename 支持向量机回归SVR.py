# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:08:17 2018

@author: zhangkefei052560
"""
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.cross_validation import train_test_split 
from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm.classes import NuSVR

#导入数据
dataset = read_csv('air_pollution.csv')
examDf = DataFrame(dataset)
new_examDf = DataFrame(examDf.drop('data',axis=1))

#将数据分割为训练集与测试集(手动分割)
X_train,X_test,y_train,y_test = (new_examDf.iloc[:220,:7],new_examDf.iloc[220:-1,:7],new_examDf.AQI[1:221],new_examDf.AQI[221:])
X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)


#设置超参数、拟合、预测
clf = NuSVR(nu=0.5, C=1.0, kernel='linear', degree=3, gamma='auto')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#误差分析,均方根误差RMSE、平均相对误差
RMSE = sqrt(mean_squared_error(y_test,y_pred))
print ('均方根误差: %.3f' % RMSE)
MAE = mean_absolute_error(y_test,y_pred)
print('平均绝对误差: %.3f' % MAE)
e = (abs((y_test - y_pred)/y_test))
print('平均相对误差：%.3f' % np.mean(e))
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("predict——test")
plt.ylabel('AQI')
plt.show()
