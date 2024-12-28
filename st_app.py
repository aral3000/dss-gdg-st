import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open("rfg.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Streamlit app title
st.title("Salary Prediction App")

# Sidebar for user input
st.sidebar.header("Input Features")

# Input fields for user to enter data
def get_user_input():
    work_year = st.sidebar.selectbox("Work Year", [2020, 2021, 2022])
    experience_level = st.sidebar.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
    employment_type = st.sidebar.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
    job_title = st.sidebar.selectbox(
        "Job Title",
        ["Data Scientist", "BI Data Analyst", "ML Engineer", "Lead Machine Learning Engineer"]
    )
    salary = st.sidebar.number_input("Salary in Local Currency", min_value=0, value=1000000)
    salary_currency = st.sidebar.selectbox("Currency", ["USD", "HUF", "CLP", "JPY", "INR"])
    employee_residence = st.sidebar.selectbox("Employee Residence", ["CL", "HU", "JP", "IN", "US"])
    remote_ratio = st.sidebar.slider("Remote Ratio", min_value=0, max_value=100, value=50)
    company_location = st.sidebar.selectbox("Company Location", ["CL", "HU", "JP", "IN", "US"])
    company_size = st.sidebar.selectbox("Company Size", ["S", "M", "L"])

    # Return the data as a DataFrame
    user_data = {
        "work_year": [work_year],
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "job_title": [job_title],
        "salary": [salary],
        "salary_currency": [salary_currency],
        "employee_residence": [employee_residence],
        "remote_ratio": [remote_ratio],
        "company_location": [company_location],
        "company_size": [company_size],
    }
    return pd.DataFrame(user_data)

# Get user input
user_input = get_user_input()

# Display user input
st.write("### User Input:")
st.write(user_input)

# Preprocess user input (You must use the same preprocessing as in model training)
def preprocess_input(input_df):
    # Perform the same preprocessing steps used during training
    # Example: One-hot encoding, scaling, etc.
    categorical_cols = ["experience_level", "employment_type", "job_title",
                        "salary_currency", "employee_residence",
                        "company_location", "company_size"]

    encoder = pickle.load(open("encoder.pkl", "rb"))  # Load encoder
    input_encoded = encoder.transform(input_df[categorical_cols])

    # Scale numerical features
    numerical_cols = ["work_year", "salary", "remote_ratio"]
    scaler = pickle.load(open("scaler.pkl", "rb"))  # Load scaler
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Merge encoded and numerical data
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    input_df = input_df.drop(columns=categorical_cols)
    input_df = pd.concat([input_df.reset_index(drop=True), input_encoded_df.reset_index(drop=True)], axis=1)

    return input_df

# Preprocess the user input
try:
    processed_input = preprocess_input(user_input)

    # Make prediction
    prediction = model.predict(processed_input)

    # Display prediction
    st.write("### Predicted Salary in USD:")
    st.write(f"${prediction[0]:,.2f}")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
