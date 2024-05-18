import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers

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
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Define the split ratio
train_ratio = 0.80
test_ratio = 0.20

# Split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_ratio, random_state=42)

# Reshape into 3D array for LSTM
sequence_length = 20  # Increased sequence length
def create_sequences(data, target, sequence_length):
    sequences = []
    targets = []
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i])
        targets.append(target[i])
    return np.array(sequences), np.array(targets)

X_train_sequences, y_train_sequences = create_sequences(X_train, y_train, sequence_length)
X_test_sequences, y_test_sequences = create_sequences(X_test, y_test, sequence_length)

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_sequences.shape[1], X_train_sequences.shape[2]), kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model
history = model.fit(X_train_sequences, y_train_sequences, epochs=50, validation_data=(X_test_sequences, y_test_sequences), shuffle=False, callbacks=[early_stopping])

# Plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Train vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
