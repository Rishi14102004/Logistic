# -*- coding: utf-8 -*-
"""
Diabetes Prediction using Logistic Regression
"""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Title
st.title('Model Deployment: Diabetes Prediction (Logistic Regression)')

# Sidebar
st.sidebar.header('User Input Parameters')

# User input function
def user_input_features():
    Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    Glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    BloodPressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
    SkinThickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    Insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=900, value=80)
    BMI = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', value=0.5)
    Age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=30)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

# Display input
st.subheader('User Input Parameters')
st.write(df)

# Load dataset
diabetes = pd.read_csv("diabetes.csv")

# Split features and target
X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, Y)

# Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Output
st.subheader('Predicted Result')
st.write('Diabetic' if prediction[0] == 1 else 'Not Diabetic')

st.subheader('Prediction Probability')
st.write(prediction_proba)
