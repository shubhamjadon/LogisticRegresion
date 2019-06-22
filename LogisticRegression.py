#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import math

#importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#dummy encoding (making seprate columns for seprate categories )
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #here we transformed X_test by fitting sc_X to X_train so that both have same scaling

#Logistic regression using gradient descent

theta = [0,0,0,0] #values of coffiecient of independent variable

#Making X[i][0] = 1 so that we get genral form like h = theta0+ theta1*x + theta2*x ... after multiplying theta and X_train matrices
i = 0
j = 0
while(i<360):
    X_train[i][j] = 1
    if i<40:
        X_test[i][j] = 1
    i += 1

#function to return  derivative of cost function that will be subtracted form each theta respectevely.
def averageCost(temp):
    total = 0
    for (i,j) in zip(X_train,Y_train):
        y_pred = h_theta(i) #calculating value of predicted y by signoid or logistic function
        total = total + (y_pred - j)*i[temp]
    return total/len(X_train)

#calculating derivative of cost function for each theta
cost = [0,0,0,0]

def updateCostMatrix():
    s = 0 #it denotes index of respective theta
    while s<len(theta):
        cost[s] = averageCost(s)
        s += 1

def updateTheta():
    i = 0
    while(i<len(theta)):
        theta[i] = theta[i] - 0.1*cost[i]
        i += 1

def h_theta(i):
    mul = np.matmul(theta,i)
    if(mul<0):
            mul = 1 - 1/(1+math.exp(mul))
    else:
        mul = 1/(1+math.exp(-mul))        
    return(mul)
        
#Here loop stops when cost becomes approximately to 0
#This loop is gradient descent algorithm
while(True):
    count = 0 #it is used to check that all cost values are zeroes
    updateCostMatrix()
    updateTheta()
    #print(cost)
    for i in cost:
        if(i < 0.5 and i>-0.5):
            count += 1
    
    if(count == 4):  #4 is no of features in X_train
        break

#Calculating error of predicted values and also storing predicted values in temp
error = 0 #initialising it.
temp = []
for i,j in zip(X_test,Y_test):
    y_pred = h_theta(i)
    if(y_pred>=0.5):
        y_pred  = 1
    else:
        y_pred = 0
    temp.append(y_pred)
    if(y_pred != j):
        error = error + 1
error = error*100/40
print("Error is:",error,"%")

