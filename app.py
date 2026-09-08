import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the geography encoder and scaler (We will bypass the broken gender pickle)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')


geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

# Explicitly hardcode text choices instead of using label_encoder_gender.classes_
gender = st.selectbox('Gender', ['Female', 'Male'])

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],  
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with display input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Display the clean table to the user on the Streamlit interface
st.write("### Input Features Overview", input_data)

# Duplicate a copy to transform categories into raw numeric values
processed_input = input_data.copy()

# Manually map text selections to expected model integers (Female=0, Male=1)
gender_mapping = {'Female': 0, 'Male': 1}
processed_input['Gender'] = processed_input['Gender'].map(gender_mapping).astype(int)

# Scale using the fully numeric processed dataframe
input_data_scaled = scaler.transform(processed_input)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
