import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import load_model

model = load_model('./Model/model.h5')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['sex', 'smoker', 'region']),
        ('num', StandardScaler(), ['age', 'bmi', 'children'])
    ])

data = pd.read_csv('insurance.csv')
X = data.drop('charges', axis=1)
preprocessor.fit(X)

def preprocess_data(data):
    return preprocessor.transform(data)

def main():
    st.title("Insurance Charges Prediction")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    income = st.number_input("Income", min_value=0, value=50000)

    prediction_button = st.button("Predict Insurance Charges")

    if prediction_button:
        new_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region],
            'income': [income]
        })
        new_data_transformed = preprocess_data(new_data)

        prediction = model.predict(new_data_transformed)[0][0]

        st.write("Predicted charges:", prediction)
        if prediction >= income / 12:
            st.write("The applicant is likely to not be able to pay for the insurance")

if __name__ == '__main__':
    main()
