#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Reading Dataset
df = pd.read_csv('google1.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,1]

#Splitting Dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediction 
y_pred = regressor.predict(X_test)


#Visualizations of Training Set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Google Stock Prediction (Training Set)')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.show()

#Visualizations of Testing Set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Google Stock Prediction (Testing Set)')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.show()