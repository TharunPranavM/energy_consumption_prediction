import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import sklearn.metrics as metrics
from math import sqrt

import keras
from keras import Sequential
from keras.layers import Dropout, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import get_custom_objects
from keras import backend as K

# Read the dataset
raw1 = pd.read_csv("D:\energy_consumption\powerconsumption.csv", header=0)
raw1['Datetime'] = pd.to_datetime(raw1['Datetime'], utc=True)
raw1 = raw1.set_index(raw1['Datetime']).drop('Datetime', axis=1)
raw1 = raw1.set_index(raw1.index.tz_convert(None) + pd.offsets.Hour(-2))

# Resample and interpolate missing values
energy_data = raw1.resample('10min').interpolate(method='linear', limit_area='inside')

# Define features (X) and target (y)
X = energy_data.drop(columns=['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']).values
y = energy_data['PowerConsumption_Zone1'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# Train Ridge regression model
model1 = Ridge()
model1.fit(X_train, y_train)
# Save model1
joblib.dump(model1, 'D:\energy_consumption/model1.pkl')

# Train Extremely Randomized Trees regressor model
model2 = ExtraTreesRegressor(max_depth=25, n_estimators=400, bootstrap=True, max_samples=0.7)
model2.fit(X_train, y_train)
# Save model2
joblib.dump(model2, 'D:\energy_consumption/model2.pkl')

# Train Random Forest regressor model
model3 = RandomForestRegressor(max_depth=20, n_estimators=400, max_features=0.9)
model3.fit(X_train, y_train)
# Save model3
joblib.dump(model3, 'D:\energy_consumption/model3.pkl')

# Train Gradient Boosting regressor model
model4 = GradientBoostingRegressor(max_depth=8, loss='squared_error', n_estimators=400)
model4.fit(X_train, y_train)
# Save model4
joblib.dump(model4, 'D:\energy_consumption/model4.pkl')

# Train Support Vector Regressor (SVR) model
model5 = SVR(kernel='rbf', C=900, epsilon=1, gamma='scale', cache_size=1000)
model5.fit(X_train, y_train)
# Save model5
joblib.dump(model5, 'D:\energy_consumption/model5.pkl')

# Define the ANN architecture
def create_model():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse', 'mape'])

    return model

# Initialize the ANN model
model6 = create_model()

# Train the ANN model
model_save_path = 'D:\energy_consumption/model6.h5'
callback_cp = ModelCheckpoint(model_save_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
callback_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
history = model6.fit(X_train, y_train, epochs=200, verbose=1, batch_size=32, validation_split=0.1, callbacks=[callback_cp, callback_es])

# Save model6
model6.save('D:\energy_consumption/model6.h5')
joblib.dump(model6, 'D:\energy_consumption/model6.pkl')

# Evaluate the ANN model
train_loss, train_mae, train_mse, train_mape = model6.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae, test_mse, test_mape = model6.evaluate(X_test, y_test, verbose=0)

print("ANN Training Loss: {:.4f}, Training MAE: {:.4f}, Training MSE: {:.4f}, Training MAPE: {:.4f}".format(train_loss, train_mae, train_mse, train_mape))
print("ANN Testing Loss: {:.4f}, Testing MAE: {:.4f}, Testing MSE: {:.4f}, Testing MAPE: {:.4f}".format(test_loss, test_mae, test_mse, test_mape))

# Combine predictions from all models
train_preds = np.column_stack([
    model1.predict(X_train),
    model2.predict(X_train),
    model3.predict(X_train),
    model4.predict(X_train),
    model5.predict(X_train),
    model6.predict(X_train)
])
test_preds = np.column_stack([
    model1.predict(X_test),
    model2.predict(X_test),
    model3.predict(X_test),
    model4.predict(X_test),
    model5.predict(X_test),
    model6.predict(X_test)
])

# Train meta-model (Random Forest)
meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
meta_model.fit(train_preds, y_train)

# Use the meta-model to make final predictions on the test set
final_predictions = meta_model.predict(test_preds)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f'Final Ensemble RMSE: {rmse:.2f}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, final_predictions)
print("Final Ensemble MAPE:", mape)
print("Final Ensemble Percentual:", metrics.mean_absolute_error(y_test, final_predictions) / y_test.mean() * 100, "%")
