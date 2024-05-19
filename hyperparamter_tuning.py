import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

# Load the dataset
dataset = pd.read_csv('Tesla.csv')

# Setting the 'Date' as the index of the DataFrame
dataset.index = pd.to_datetime(dataset['Date'])
dataset.drop(columns=['Date'], inplace=True)
dataset.drop(columns=['Unnamed: 0'], inplace=True)

# Normalizing the data
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset)

# Function to create model
def create_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(80, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'batch_size': [16, 24, 32],
    'epochs': [75, 100, 125],
}

# Wrap Keras model in a scikit-learn estimator
keras_regressor = KerasRegressor(build_fn=create_model, verbose=0)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=keras_regressor, param_distributions=param_dist, n_iter=10, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', verbose=2)

# Splitting the data into features and target
X = []
y = []
sequence_length = 80
for i in range(sequence_length, len(dataset_scaled)):
    X.append(dataset_scaled[i-sequence_length:i, 0])
    y.append(dataset_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Fit the model
random_search.fit(X, y)

# Print best parameters and best score
print("Best parameters found: ", random_search.best_params_)
print("Best RMSE found: ", np.sqrt(np.abs(random_search.best_score_)))
