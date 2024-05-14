import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

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

# Fill missing values with the mean of each column
dataset = dataset.fillna(dataset.mean())

# Split the dataset into features (X) and target (y)
X = dataset.drop('CO(GT)', axis=1)
y = dataset['CO(GT)']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape the data to be 3D
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
param_grid = dict(optimizer=optimizer)

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
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Plot the training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(y_test, color='blue', label='Actual')
plt.plot(model.predict(X_test), color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test, verbose=2)
print('\nTest loss:', test_loss)