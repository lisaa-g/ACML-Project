import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
dataset = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

# Drop the last empty column
dataset = dataset.dropna(axis=1, how='all')

# Convert date and time to datetime format and set it as index
dataset['DateTime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'], format='%d/%m/%Y %H.%M.%S')
dataset = dataset.set_index('DateTime')

# Drop the original date and time columns
dataset = dataset.drop(['Date', 'Time'], axis=1)

# Replace -200 (which seems to be a placeholder for missing values) with NaN
dataset = dataset.replace(-200, np.nan)

# Fill missing values with the forward fill method
dataset = dataset.fillna(method='ffill')

# Split the dataset into features (X) and target (y)
X = dataset.drop('CO(GT)', axis=1)
y = dataset['CO(GT)']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape the data to be 3D
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Split the dataset into training, validation and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Define a function that creates the model (required for KerasClassifier)
# while accepting the hyperparameters we want to tune
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(SimpleRNN(32, return_sequences=True, input_shape=(None, X_train.shape[-1])))
    model.add(SimpleRNN(32))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Create a KerasRegressor instance
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_grid = dict(optimizer=optimizer, epochs=epochs)

# Conduct Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print out the results of hyperparameter tuning
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"Means: {mean}, Stdev: {stdev} with: {param}")

# Train the model using the best parameters found
model = create_model(optimizer=grid_result.best_params_['optimizer'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=grid_result.best_params_['epochs'], batch_size=32)

# Plot the training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.ylabel('Loss Value')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(y_test, color='blue', label='Actual')
plt.plot(model.predict(X_test), color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.ylabel('Target Variable Value')
plt.xlabel('Sample Index')
plt.legend()

plt.show()

# Predict the training data
y_train_pred = model.predict(X_train)

# Calculate metrics
train_loss = model.evaluate(X_train, y_train, verbose=0)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
test_loss = model.evaluate(X_test, y_test, verbose=2)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print Training metrics
print('\nTrain loss:', train_loss)
print('Train Mean Absolute Error:', mae_train)
print('Train Root Mean Squared Error:', rmse_train)

# Print Test metrics
print('\nTest loss:', test_loss)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)