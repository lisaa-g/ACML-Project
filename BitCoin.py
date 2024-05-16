# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Reading CSV file from train set
training_set = pd.read_csv('BitCoin_Data.csv')

# Selecting the second column [for prediction]
training_set = training_set.iloc[:,1:2].values

# Scaling of Data [Normalization]
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Creating training data
X_train = training_set[0:897]
Y_train = training_set[1:898]

# Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
X_train = np.reshape(X_train, (897, 1, 1))

# RNN Layers
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the Recurrent Neural Network
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the Recurrent Neural Network [epochs is a kind of number of iteration]
regressor.fit(X_train, Y_train, batch_size = 32, epochs = 200)

# Reading CSV file from test set
test_set = pd.read_csv('BitCoin_Data.csv')

# Selecting the second column from test data 
real_btc_price = test_set.iloc[:,1:2].values      

# Getting the predicted BTC value of the first week of Dec 2017  
inputs = sc.transform(real_btc_price)

# Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
inputs = np.reshape(inputs, (inputs.shape[0], 1, 1))
predicted_btc_price = regressor.predict(inputs)
predicted_btc_price = sc.inverse_transform(predicted_btc_price)

# Graphs for predicted values
plt.plot(real_btc_price, color = 'red', label = 'Real BTC Value')
plt.plot(predicted_btc_price, color = 'blue', label = 'Predicted BTC Value')
plt.title('BTC Value Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Value')
plt.legend()
plt.show()