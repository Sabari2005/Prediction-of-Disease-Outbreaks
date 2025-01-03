import streamlit as st
import pickle
import numpy as np
import os
from groq import Groq

# Function to load models
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load trained models
diabetes_model = load_model('diabetes_model.pkl')
heart_disease_model = load_model('heart_disease_model.pkl')
parkinsons_model = load_model('parkinsons_disease_model.pkl')

# Function to predict diabetes
def predict_diabetes(inputs):
    return diabetes_model.predict(np.array([inputs]))

# Function to predict heart disease
def predict_heart_disease(inputs):
    return heart_disease_model.predict(np.array([inputs]))

# Function to predict Parkinson's Disease
def predict_parkinsons(inputs):
    return parkinsons_model.predict(np.array([inputs]))

# Setting up the Streamlit interface
st.title("Disease Prediction and Chat Interface")

# Tabs for the models
tab1, tab2, tab3, tab4 = st.tabs(["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Disease Prediction", "Chat Interface"])

# Tab 1: Diabetes Prediction
with tab1:
    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", value=0, key="diabetes_pregnancies")
    glucose = st.number_input("Glucose", value=0, key="diabetes_glucose")
    blood_pressure = st.number_input("Blood Pressure", value=0, key="diabetes_blood_pressure")
    skin_thickness = st.number_input("Skin Thickness", value=0, key="diabetes_skin_thickness")
    insulin = st.number_input("Insulin", value=0, key="diabetes_insulin")
    bmi = st.number_input("BMI", value=0.0, key="diabetes_bmi")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", value=0.0, key="diabetes_pedigree")
    age = st.number_input("Age", value=0, key="diabetes_age")

    if st.button("Predict Diabetes", key="predict_diabetes"):
        prediction = predict_diabetes([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])
        st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")

# Tab 2: Heart Disease Prediction
with tab2:
    st.header("Heart Disease Prediction")

    age = st.number_input("Age", value=0, key="heart_age")
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female", key="heart_sex")
    cp = st.number_input("Chest Pain Type", value=0, key="heart_cp")
    trestbps = st.number_input("Resting Blood Pressure", value=0, key="heart_trestbps")
    chol = st.number_input("Cholesterol", value=0, key="heart_chol")
    fbs = st.number_input("Fasting Blood Sugar", value=0, key="heart_fbs")
    restecg = st.number_input("Rest ECG", value=0, key="heart_restecg")
    thalach = st.number_input("Maximum Heart Rate", value=0, key="heart_thalach")
    exang = st.number_input("Exercise Induced Angina", value=0, key="heart_exang")
    oldpeak = st.number_input("Oldpeak", value=0.0, key="heart_oldpeak")
    slope = st.number_input("Slope", value=0, key="heart_slope")
    ca = st.number_input("Number of Major Vessels", value=0, key="heart_ca")
    thal = st.number_input("Thal", value=0, key="heart_thal")

    if st.button("Predict Heart Disease", key="predict_heart"):
        prediction = predict_heart_disease([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        st.write("Prediction:", "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")

# Tab 3: Parkinson's Disease Prediction
with tab3:
    st.header("Parkinson's Disease Prediction")

    inputs = []
    input_names = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE"
    ]

    for i, name in enumerate(input_names):
        value = st.number_input(name, value=0.0, key=f"parkinsons_{i}")
        inputs.append(value)

    if st.button("Predict Parkinson's Disease", key="predict_parkinsons"):
        prediction = predict_parkinsons(inputs)
        st.write("Prediction:", "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's")

# Tab 4: Chat Interface
with tab4:
    st.header("Chat Interface")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User input
    user_input = st.text_input("Ask a question about your health:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Set up Groq API client
        client = Groq(api_key="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")

        # Generate chat completion
        chat_completion = client.chat.completions.create(
            messages=st.session_state.messages,
            model="llama-3.3-70b-versatile",
        )

        response_content = chat_completion.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")
