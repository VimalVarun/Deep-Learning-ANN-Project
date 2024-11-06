import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the trained model
model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Load the saved weights
model.load_weights("model_weights.h5")

# Load the encoders and scaler
try:
    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("OneHot_Encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load one or more encoders or scaler: {e}")

# Streamlit app title
st.title("Customer Churn Prediction")

# Collect user inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0] if 'onehot_encoder_geo' in locals() else [])
gender = st.selectbox("Gender", label_encoder_gender.classes_ if 'label_encoder_gender' in locals() else [])
age = st.slider("Age", 18, 95)
balance = st.number_input("Balance", min_value=0.0, format="%.2f")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, format="%.2f")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0] if 'label_encoder_gender' in locals() else 0],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode geography
if 'onehot_encoder_geo' in locals():
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))
    # Combine with the main input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
if 'scaler' in locals():
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    if 'model' in locals():
        prediction = model.predict(input_data_scaled)
        prediction_prob = prediction[0][0]

        # Display the prediction result
        if prediction_prob > 0.5:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is not likely to churn.")
    else:
        st.error("Model is not available for predictions.")
else:
    st.error("Scaler is not available for data preprocessing.")