from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
california=fetch_california_housing()

X_train,X_test,Y_train,Y_test=train_test_split(california.data,california.target,test_size=0.2)

# normalize the data
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# train the model

model=LinearRegression()
model.fit(X_train,Y_train)

output=model.predict(X_test)
mse=mean_squared_error(output,Y_test)
print(mse)

#MLP
import keras 
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from keras.utils import to_Catagorical
import numpy as np
import matplotlib.pyplot as plt

# load the data set
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
# print(X_train.shape,Y_test.shape)
