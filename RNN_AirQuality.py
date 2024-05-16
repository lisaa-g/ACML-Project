import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the CSV file
dataset = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

# Drop the last empty column
dataset = dataset.dropna(axis=1, how='all')

# Replace periods in 'Time' with colons
dataset['Time'] = dataset['Time'].str.replace('.', ':', regex=False)

# Convert 'Date' and 'Time' to datetime format and set it as index
dataset['DateTime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'])
dataset.set_index('DateTime', inplace=True)

# Drop the original date and time columns
dataset = dataset.drop(['Date', 'Time'], axis=1)

# Replace -200 (which seems to be a placeholder for missing values) with NaN
dataset = dataset.replace(-200, np.nan)

# Fill missing values with the forward fill method
dataset = dataset.fillna(method='ffill')

# Save the DataFrame to a new CSV file with the index
dataset.to_csv('new_AirQualityUCI.csv', index=True)

# Selecting the 2nd column for prediction
# Split the dataset into features (X) and target (y)
X = dataset.drop('CO(GT)', axis=1)
y = dataset['CO(GT)']

# Scaling of Data [Normalization]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = np.array(y).reshape(-1, 1)  # Reshape y to a 2D array
y_scaled = scaler.fit_transform(y)

# Define the split ratio
train_ratio = 0.80
test_ratio = 0.20

# Split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_ratio, random_state=42)

# Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Recurrent Neural Network Layers
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, X_train.shape[2])))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the Recurrent Neural Network
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the Recurrent Neural Network to the Training set
history = regressor.fit(X_train, y_train, batch_size=32, epochs=50)

# Getting the predicted CO(GT) value
predicted_CO_GT = regressor.predict(X_test)
predicted_CO_GT = scaler.inverse_transform(predicted_CO_GT)

# Graphs for predicted values
plt.plot(scaler.inverse_transform(y_test), color='red', label='Real CO(GT) Value')
plt.plot(predicted_CO_GT, color='blue', label='Predicted CO(GT) Value')
plt.title('CO(GT) Value Prediction')
plt.xlabel('Days')
plt.ylabel('CO(GT) Value')
plt.legend()
plt.show()

# Plot the training loss
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Print out the final training loss
print(f"Final training loss: {history.history['loss'][-1]}")
