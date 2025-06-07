import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define expected feature columns based on what the model was trained on
expected_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
                 'average_montly_hours', 'time_spend_company', 'Work_accident',
                 'promotion_last_5years', 'low', 'medium']

# Streamlit input UI
st.title("Employee Retention Predictor")

satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_monthly_hours = st.number_input("Average Monthly Hours", 80, 320, 160)
time_spent = st.number_input("Time Spent at Company (years)", 1, 10, 3)
work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

# Create input DataFrame
input_dict = {
    'satisfaction_level': [satisfaction_level],
    'last_evaluation': [last_evaluation],
    'number_project': [number_project],
    'average_montly_hours': [average_monthly_hours],
    'time_spend_company': [time_spent],
    'Work_accident': [work_accident],
    'promotion_last_5years': [promotion_last_5years],
    'salary': [salary]
}
input_data = pd.DataFrame(input_dict)

# One-hot encode salary and reindex to match training columns
input_data = pd.get_dummies(input_data, columns=['salary'])
input_data = input_data.reindex(columns=expected_cols, fill_value=0)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("This employee is likely to leave.")
    else:
        st.success("This employee is likely to stay.")
