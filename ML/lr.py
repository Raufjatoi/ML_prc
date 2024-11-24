import pandas as p 
import math as m 
import numpy as n 
from sklearn import preprocessing , svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = p.read_csv('google.csv')
data = data[['adjOpen', 'adjHigh', 'adjLow', 'adjClose' , 'adjVolume']]
data['HL_PCT'] = (data['adjHigh'] - data['adjClose']) / data['adjClose'] * 100.0
data['PCT_change'] = (data['adjClose'] - data['adjOpen']) / data['adjOpen'] * 100.0

data = data[['adjClose', 'HL_PCT', 'PCT_change' , 'adjVolume']]

forecast_col = 'adjClose'
data.fillna(-9999, inplace=True)

forecast_out = int(m.ceil(0.01*len(data)))
print(forecast_out)

data['label'] = data[forecast_col].shift(-forecast_out)

data.dropna(inplace=True)
#print(data.head())
#print(data.tail())

X = n.array(data.drop(['label'],axis=1))
y = n.array(data['label'])
X = preprocessing.scale(X)
y = n.array(data['label'])
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf =  LinearRegression(n_jobs=10) #svm SVR(kernel='poly)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)