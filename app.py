import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model and scaler
with open('churn.pkl', 'rb') as f:
    model = pickle.load(f)
with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Title and description
st.title("Customer Churn Prediction")
st.write("Enter the customer details below to predict the likelihood of churn.")

# Input fields for customer details
Age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
Subscription_Length_Months = st.number_input("Subscription Length (Months)", min_value=1, max_value=60, step=1)
Monthly_Bill = st.number_input("Monthly Bill ($)", min_value=0.0, step=1.0)
Total_Usage_GB = st.number_input("Total Usage (GB)", min_value=0.0, step=0.1)

# Location options with dummy variables for encoding
location = st.selectbox("Location", ["Houston", "Los Angeles", "Miami", "New York","Chicago"])
Location_Houston = 1 if location == "Houston" else 0
Location_Los_Angeles = 1 if location == "Los Angeles" else 0
Location_Miami = 1 if location == "Miami" else 0
Location_New_York = 1 if location == "New York" else 0
location_Chicago = 1 if location == "Chicago" else 0

# Convert gender to dummy variable
Gender_Male = 1 if gender == "Male" else 0

# Collect all input features in the correct order
features = np.array([[Age,Subscription_Length_Months, Monthly_Bill, Total_Usage_GB,Gender_Male,
                     Location_Los_Angeles,Location_Miami, Location_New_York,Location_Houston]])



# Scale the features using the loaded StandardScaler
scaled_features = scaler.transform(features)

# Button to make prediction
if st.button("Predict Churn"):
    # Make prediction using the scaled features
    prediction = model.predict(scaled_features)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    
    # Display the result
    st.write(f"Prediction: {result}")
