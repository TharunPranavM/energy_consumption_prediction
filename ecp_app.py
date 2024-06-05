import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained models
model1 = joblib.load("D:\energy_consumption\model1.pkl")  # Ridge
model2 = joblib.load("D:\energy_consumption\model2.pkl")  # ExtraTreesRegressor
model3 = joblib.load("D:\energy_consumption\model3.pkl")  # RandomForestRegressor
model4 = joblib.load("D:\energy_consumption\model4.pkl")  # GradientBoostingRegressor
model5 = joblib.load("D:\energy_consumption\model5.pkl")  # SVR
model6 = joblib.load("D:\energy_consumption\model6.pkl")  # ANN

def predict_power_consumption(temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows):
    # Combine input features
    X = np.array([[temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows]])

    # Make predictions using all models
    pred1 = model1.predict(X).item()
    pred2 = model2.predict(X).item()
    pred3 = model3.predict(X).item()
    pred4 = model4.predict(X).item()
    pred5 = model5.predict(X).item()
    pred6 = model6.predict(X).item()

    # Final prediction using meta-model (Random Forest)
    final_prediction = np.mean([pred1, pred2, pred3, pred4, pred5, pred6])

    return final_prediction


# Streamlit UI
st.title('Power Consumption Prediction')
st.sidebar.header('Input Parameters')

temperature = st.sidebar.slider('Temperature', min_value=-10.0, max_value=40.0, value=20.0, step=0.1)
humidity = st.sidebar.slider('Humidity', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
wind_speed = st.sidebar.slider('Wind Speed', min_value=0.0, max_value=30.0, value=10.0, step=0.1)
general_diffuse_flows = st.sidebar.slider('General Diffuse Flows', min_value=0.0, max_value=10000.0, value=5000.0, step=10.0)
diffuse_flows = st.sidebar.slider('Diffuse Flows', min_value=0.0, max_value=10000.0, value=5000.0, step=10.0)

if st.sidebar.button('Predict'):
    power_consumption = predict_power_consumption(temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows)
    st.write('Predicted Power Consumption:', round(power_consumption, 2), 'kWh')
