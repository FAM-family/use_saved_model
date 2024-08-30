import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

st.title("Gender Prediction App")

# Input fields for user to enter values
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
#race = st.selectbox("Race", options=[1, 2, 3, 4, 5])  # Adjust options as needed

# Predict button
if st.button("Predict Gender"):
    # Prepare the input data
    input_data = np.array([[height, weight, age]])
  #, race]])
    
    # Make prediction
    prediction = model.predict(input_data)

    # Alternatively, use predict_proba if you need probabilities
    prob = model.predict_proba(input_data)

    st.write(f"---")
    
    # Display the result
    gender = "Male" if prediction[0] == 0 else "Female"
    st.write(f"The predicted gender is: {gender}")
    
    st.write(f"---")
    st.write(f"Probabilities for each gender:")
    st.write(f"{prob}")


