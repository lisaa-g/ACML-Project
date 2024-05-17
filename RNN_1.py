import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from keras import Sequential, Input
from keras.layers import Dense, LSTM, Dropout
from scikeras.wrappers import KerasRegressor

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

# Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Function to create model, required for KerasRegressor
def create_model(units=50, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=units, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrap the Keras model with KerasRegressor
model = KerasRegressor(
    model=create_model,
    verbose=0
)

# Define the hyperparameter grid
param_grid = {
    'model__units': [50, 100, 150],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

# Cross-validation and hyperparameter tuning
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=kfold, random_state=42, verbose=1)

# Fit the model
random_search_result = random_search.fit(X_train, y_train)

# Best parameters and best model
best_params = random_search_result.best_params_
best_model = random_search_result.best_estimator_

# Evaluate the best model on the test set
test_loss = best_model.score(X_test, y_test)
print(f"Best parameters: {best_params}")
print(f"Test loss: {test_loss}")

# Predict and invert the scaling
predicted_CO_GT = best_model.predict(X_test)
predicted_CO_GT = scaler.inverse_transform(predicted_CO_GT.reshape(-1, 1))

# Plotting
plt.plot(scaler.inverse_transform(y_test), color='red', label='Real CO(GT) Value')
plt.plot(predicted_CO_GT, color='blue', label='Predicted CO(GT) Value')
plt.title('CO(GT) Value Prediction')
plt.xlabel('Days')
plt.ylabel('CO(GT) Value')
plt.legend()
plt.show()

# Print out the final training and test loss
history = best_model.history_
print(f"History keys: {history.keys()}")  # Print the history keys to debug
plt.figure(figsize=(12, 4))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.axhline(test_loss, color='red', label='Test Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

print(f"Final training loss: {history['loss'][-1]}")
print(f"Test loss: {test_loss}")
