import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import load_model

model = load_model('./Model/model.h5')

df=pd.read_csv('./insurance.csv')
# st.dataframe(df)

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
    st.title("Insurance Premium Prediction")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    height = st.number_input("Height (in m)", min_value=100, value=180)
    weight = st.number_input("Weight (in kg)", min_value=50, value=70)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    income = st.number_input("Income (in ₹)", min_value=0, value=50000)

    prediction_button = st.button("Predict Insurance Charges")

    bmi = weight/((height/100)*(height/100))

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

    st.markdown("#### BMI stats")
    st.write("BMI: ",bmi)
    if bmi >=18.5 and bmi<=24.9:
        st.write("Your BMI falls in the healthy weight range")
    elif bmi>=25.0 and bmi<=29.9:
        st.write("Your BMI falls in the overweight range")
    elif bmi>=30.0:
        st.write("Your BMI falls in the obese weight range")
    elif bmi<18.5:
        st.write("Your BMI falls in the underweight range")

    st.markdown("#### Learn how these factors that affect your premium \n  ##### Smoking")  

    smoker_data=df.groupby('smoker')['charges'].mean()
    st.bar_chart(smoker_data)
    smoker_average=df[df['smoker']=="yes"]['charges'].mean()
    non_smoker_average=df[df['smoker']=="no"]['charges'].mean()

    st.write("Premiums for smokers are on average {:,.2f}".format(1+(smoker_average-non_smoker_average)/smoker_average),"times higher than non-smokers")    
    st.markdown("#### BMI")
    bmi_data=df.groupby('bmi')['charges'].mean()
    st.bar_chart(bmi_data)
    
    st.markdown("#### Region")
    region_data=df.groupby('region')['charges'].mean()
    st.bar_chart(region_data)
    
if __name__ == '__main__':
    main()
