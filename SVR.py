
"""
Created on Thu Oct  4 16:08:17 2018

"""
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.cross_validation import train_test_split 
from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm.classes import NuSVR


dataset = read_csv('air_pollution.csv')
examDf = DataFrame(dataset)
new_examDf = DataFrame(examDf.drop('data',axis=1))


X_train,X_test,y_train,y_test = (new_examDf.iloc[:220,:7],new_examDf.iloc[220:-1,:7],new_examDf.AQI[1:221],new_examDf.AQI[221:])
X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)



clf = NuSVR(nu=0.5, C=1.0, kernel='linear', degree=3, gamma='auto')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


RMSE = sqrt(mean_squared_error(y_test,y_pred))
print ('RMSE: %.3f' % RMSE)
MAE = mean_absolute_error(y_test,y_pred)
print('MAE: %.3f' % MAE)
e = (abs((y_test - y_pred)/y_test))
print('mBA：%.3f' % np.mean(e))
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend(loc="upper right")
plt.xlabel("predict——test")
plt.ylabel('AQI')
plt.show()
