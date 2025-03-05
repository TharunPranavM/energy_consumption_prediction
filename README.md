
---

# **Energy Consumption Prediction**

This project predicts power consumption using multiple machine learning models, including Ridge Regression, Random Forest, Gradient Boosting, SVR, and an Artificial Neural Network (ANN). The predictions are combined using an ensemble meta-model for improved accuracy. A Streamlit-based web application provides an interactive user interface for making predictions.

## **Table of Contents**
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works (Workflow)](#how-it-works-workflow)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Web Application](#web-application)
- [Results](#results)
- [Contributors](#contributors)

---

## **Overview**
The goal of this project is to develop a predictive model for energy consumption based on weather-related features. The model is trained on historical power consumption data and employs various machine learning algorithms to enhance accuracy through an ensemble approach.

---

## **Project Structure**
```
├── consumption_model.py     # Training and saving ML models
├── ecp_app.py               # Streamlit application for predictions
├── models/                  # Folder containing trained models
│   ├── model1.pkl           # Ridge Regression model
│   ├── model2.pkl           # Extra Trees Regressor
│   ├── model3.pkl           # Random Forest Regressor
│   ├── model4.pkl           # Gradient Boosting Regressor
│   ├── model5.pkl           # Support Vector Regressor (SVR)
│   ├── model6.pkl           # ANN model
│   ├── meta_model.pkl       # Final ensemble model (Random Forest)
└── dataset/
    ├── powerconsumption.csv  # Raw dataset used for training
```

---


## **Usage**
### **1. Train Models**
Run the `consumption_model.py` script to train the models and save them:
```bash
python consumption_model.py
```

### **2. Start the Streamlit App**
Run the `ecp_app.py` script to launch the web-based prediction tool:
```bash
streamlit run ecp_app.py
```
Once started, you can interact with the app via the provided sliders and obtain power consumption predictions.

---

## **How It Works (Workflow)**
The workflow consists of the following steps:

### **1. Data Preprocessing**
- The dataset (`powerconsumption.csv`) is loaded.
- The `Datetime` column is converted to a time-series index.
- Missing values are handled using **linear interpolation**.
- Data is resampled to 10-minute intervals to maintain consistency.

### **2. Feature Selection & Splitting**
- Features like **temperature, humidity, wind speed, general diffuse flows, and diffuse flows** are selected.
- The target variable is **PowerConsumption_Zone1**.
- The dataset is split into **70% training and 30% testing**.

### **3. Model Training**
- Multiple machine learning models are trained, including:
  - **Ridge Regression**
  - **Extra Trees Regressor**
  - **Random Forest Regressor**
  - **Gradient Boosting Regressor**
  - **Support Vector Regressor (SVR)**
  - **Artificial Neural Network (ANN)**
- The trained models are saved as `.pkl` files.

### **4. Ensemble Learning (Meta-Model)**
- Predictions from all models are combined.
- A **Random Forest meta-model** is trained on these predictions to improve accuracy.
- The final predictions are derived using the meta-model.

### **5. Web Application for User Interaction**
- A **Streamlit-based UI** allows users to input environmental factors.
- Predictions are made in real-time using the trained models.
- The ensemble model provides a final power consumption estimate.

---

## **Models Used**
The project uses the following models:
1. **Ridge Regression**
2. **Extra Trees Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **Support Vector Regressor (SVR)**
6. **Artificial Neural Network (ANN)**
7. **Meta-Model (Random Forest) for ensemble prediction**

---

## **Dataset**
- **Source**: The dataset `powerconsumption.csv` contains historical power consumption data.
- **Features**: Temperature, Humidity, Wind Speed, General Diffuse Flows, Diffuse Flows.
- **Target Variable**: `PowerConsumption_Zone1` (kWh)

---

## **Evaluation Metrics**
The models are evaluated using:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Mean Absolute Error (MAE)**

---

## **Web Application**
The project includes a Streamlit-based web application that provides an intuitive user interface for predicting power consumption based on environmental factors.

Features
✅ User-Friendly Interface – Simple sliders allow users to input environmental variables easily.
✅ Real-Time Predictions – The app instantly computes power consumption predictions using trained models.
✅ Ensemble Learning – The final prediction is an aggregation of multiple models to ensure higher accuracy.
✅ Interactive & Responsive – Users can dynamically adjust input parameters and observe changes in predictions.

---

## **Results**
The final ensemble model improves accuracy by leveraging multiple individual model predictions. The Random Forest meta-model significantly reduces error rates.

Example Performance:
- **Final Ensemble RMSE**: *3958.64*
- **Final Ensemble MAPE**: *0.11789913171750589*
- **Final Ensemble Percentual**: *10.605936147993877 %*

---
