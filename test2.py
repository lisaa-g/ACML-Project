import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import layers, callbacks
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('Tesla.csv')

# Setting the 'Date' as the index of the DataFrame
dataset.index = pd.to_datetime(dataset['Date'])
dataset.drop(columns=['Date'], inplace=True)
dataset.drop(columns=['Unnamed: 0'], inplace=True)

# Splitting the data into training and testing sets
train = dataset[dataset.index < '2021-10-08']
test = dataset[dataset.index >= '2021-10-08']

# Normalizing the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train['Close'].values.reshape(-1, 1))

# Hyperparameters
sequence_length = 80
batch_size = 24
epochs = 50
units = 50
dropout_rate = 0.2

# Creating sequences
X_train, y_train = [], []
for i in range(sequence_length, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-sequence_length:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Splitting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Reshaping for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# Building the model
model = Sequential([
    LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(dropout_rate),
    LSTM(units=units, return_sequences=True),
    Dropout(dropout_rate),
    LSTM(units=units, return_sequences=True),
    Dropout(dropout_rate),
    LSTM(units=units),
    Dropout(dropout_rate),
    Dense(units=1),
])

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])


# Training the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Plotting the training history
history_frame = pd.DataFrame(history.history)
plt.figure(figsize=(14, 8))
plt.plot(history_frame['loss'], label='Training Loss')
plt.plot(history_frame['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Preparing test data
dataset_total = dataset['Close']
inputs = dataset_total[len(dataset_total) - len(test['Close'].values) - sequence_length:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(sequence_length, len(inputs)):
    X_test.append(inputs[i-sequence_length:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predicting stock prices
best_model = model
predicted_stock_price = best_model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Evaluating the model
predictions = pd.DataFrame()
predictions['Actuals'] = test['Close'].values
predictions['Predictions'] = predicted_stock_price

# Plotting predictions vs actuals
plt.figure(figsize=(14, 8))
plt.plot(predictions['Actuals'], label='Actuals')
plt.plot(predictions['Predictions'], label='Predictions')
plt.title('Tesla Close Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
plt.show()

# R^2 score
score = r2_score(predictions['Actuals'], predictions['Predictions'])
print(f'R^2 score: {score}')

# Error Analysis
mse = mean_squared_error(predictions['Actuals'], predictions['Predictions'])
mae = mean_absolute_error(predictions['Actuals'], predictions['Predictions'])
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plotting errors
errors = predictions['Actuals'] - predictions['Predictions']
plt.figure(figsize=(14, 8))
plt.hist(errors, bins=50, alpha=0.75)
plt.title('Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

