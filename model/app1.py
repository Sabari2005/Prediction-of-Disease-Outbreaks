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
diabetes_model = load_model('model/diabetes_model.pkl')
heart_disease_model = load_model('model/heart_disease_model.pkl')
parkinsons_model = load_model('model/parkinsons_disease_model.pkl')

# Prediction functions
def predict_diabetes(inputs):
    return diabetes_model.predict(np.array([inputs]))

def predict_heart_disease(inputs):
    return heart_disease_model.predict(np.array([inputs]))

def predict_parkinsons(inputs):
    return parkinsons_model.predict(np.array([inputs]))

# Streamlit setup
st.set_page_config(page_title="Disease Prediction and Chat", layout="wide")

# Sidebar for chat history
st.sidebar.title("Chat History")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.sidebar.write(f"{'You' if message['role'] == 'user' else 'Assistant'}: {message['content']}")

# Tabs for predictions and chat
tab1, tab2, tab3, tab4 = st.tabs([
    "Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Disease Prediction", "Chat Interface"
])

# Diabetes Prediction Tab
with tab1:
    st.header("Diabetes Prediction")
    diabetes_inputs = {
        "Pregnancies": 0, "Glucose": 0, "Blood Pressure": 0, "Skin Thickness": 0, 
        "Insulin": 0, "BMI": 0.0, "Diabetes Pedigree Function": 0.0, "Age": 0
    }
    for key in diabetes_inputs:
        diabetes_inputs[key] = st.number_input(key, value=diabetes_inputs[key], key=f"diabetes_{key.replace(' ', '_').lower()}")
    if st.button("Predict Diabetes", key="predict_diabetes"):
        prediction = predict_diabetes(list(diabetes_inputs.values()))
        st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")

# Heart Disease Prediction Tab
with tab2:
    st.header("Heart Disease Prediction")
    heart_inputs = {
        "Age": 0, 
        "Sex": st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", key="heart_sex"), 
        "Chest Pain Type": 0, 
        "Resting Blood Pressure": 0, 
        "Cholesterol": 0, 
        "Fasting Blood Sugar": 0, 
        "Rest ECG": 0, 
        "Maximum Heart Rate": 0, 
        "Exercise Induced Angina": 0, 
        "Oldpeak": 0.0, 
        "Slope": 0, 
        "Number of Major Vessels": 0, 
        "Thal": 0
    }

    # Render all inputs except "Sex" using st.number_input
    for key, default in heart_inputs.items():
        if key != "Sex":
            heart_inputs[key] = st.number_input(
                key, 
                value=default, 
                step=0.1 if isinstance(default, float) else 1, 
                key=f"heart_{key.replace(' ', '_').lower()}"
            )
    
    # Prediction button
    if st.button("Predict Heart Disease", key="predict_heart"):
        prediction = predict_heart_disease(list(heart_inputs.values()))
        st.write("Prediction:", "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")

# Parkinson's Disease Prediction Tab
with tab3:
    st.header("Parkinson's Disease Prediction")
    parkinsons_inputs = []
    parkinsons_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    
    for feature in parkinsons_features:
        parkinsons_inputs.append(st.number_input(
            feature, 
            value=0.0, 
            step=0.01,  # Explicitly set step size for floating-point precision
            key=f"parkinsons_{feature}"
        ))
    
    if st.button("Predict Parkinson's Disease", key="predict_parkinsons"):
        prediction = predict_parkinsons(parkinsons_inputs)
        st.write("Prediction:", "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's")

with tab4:
    # Callback function to handle message submission
    def handle_message_submission():
        user_message = st.session_state.chat_input.strip()
        if user_message:  # Ensure the message is not empty
            # Add user message to the session state
            st.session_state.messages.append({"role": "user", "content": user_message})

            # Process response from the assistant
            try:
                # Set up Groq API client
                client = Groq(api_key="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")

                # Generate chat completion
                chat_completion = client.chat.completions.create(
                    messages=st.session_state.messages,
                    model="llama-3.3-70b-versatile",
                )

                response_content = chat_completion.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

            # Clear the input box
            st.session_state.chat_input = ""

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # Chat container
    st.markdown("""
        <style>
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            margin-bottom: 50px;
        }
        .user-message {
            text-align: right;
            margin: 10px 0;
            color: #ffffff;
            background-color: #474643;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
        }
        .assistant-message {
            text-align: left;
            margin: 10px 0;
            background-color: #474643;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
        }
        .fixed-input-box {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #ffffff;
            padding: 10px;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input box with `on_change` callback
    st.text_input(
        "Type your message here...",
        key="chat_input",
        placeholder="Type your message and press Enter...",
        label_visibility="collapsed",
        on_change=handle_message_submission
    )
