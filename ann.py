import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('sandp500.csv')
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values
DateX = dataset.iloc[:,1:5].values

Datey = dataset.iloc[:,5].values


#Test set and train set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
DateX = sc.fit_transform(DateX)


import keras
from keras.models import Sequential
from keras.layers import Dense

#ANN Initialization

classifier = Sequential()

#Adding the input layer
classifier.add(Dense(3,kernel_initializer = 'normal', activation = 'relu',input_dim = 4 ))

#Another Layer
classifier.add(Dense(3,kernel_initializer = 'normal', activation = 'relu'))

#Output Layer

classifier.add(Dense(1,kernel_initializer = 'normal'))

#Compliing

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

classifier.fit(X_train,y_train,batch_size = 10 ,nb_epoch = 100)


y_pred = classifier.predict(X_test)

y_predtrain = classifier.predict(X_train)

y_preddate = classifier.predict(DateX)
i = 0
plotx = []
for i in range(17446):
    plotx.append(i)
  
plt.plot(plotx,Datey,color = 'blue')
plt.plot(plotx,y_preddate,color = 'red')

plt.title('S&P 500 Prediction(Training Set)')
plt.show()
