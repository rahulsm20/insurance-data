import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import load_model

model = load_model('./Model/model.h5')

df=pd.read_csv('./insurance.csv')

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


    st.markdown("### Learn how these factors that affect your premium \n  #### Smoking")  

    smoker_data=df.groupby('smoker')['charges'].mean()
    st.bar_chart(smoker_data)
    smoker_average=df[df['smoker']=="yes"]['charges'].mean()
    non_smoker_average=df[df['smoker']=="no"]['charges'].mean()

    st.write("* Premiums for smokers are on average {:,.2f}".format(1+(smoker_average-non_smoker_average)/smoker_average),"times higher than non-smokers")    
    st.markdown("#### BMI")
    bmi_data=df.groupby('bmi')['charges'].mean()
    st.bar_chart(bmi_data)

    st.write("Your BMI: {:,.2f}".format(bmi))
    mean_bmi=df['bmi'].mean()
    high_bmi=df[df['bmi']>mean_bmi]['charges'].mean()
    low_bmi=df[df['bmi']<mean_bmi]['charges'].mean()
    
    
    st.write("* Average BMI : {:,.2f}".format(mean_bmi))

    if bmi >=18.5 and bmi<=24.9:
        st.write("* Your BMI falls in the healthy weight range")
    elif bmi>=25.0 and bmi<=29.9:
        st.write("* Your BMI falls in the overweight range")
    elif bmi>=30.0:
        st.write("* Your BMI falls in the obese weight range")
    elif bmi<18.5:
        st.write("* Your BMI falls in the underweight range")

    st.write("* Premiums for people with BMI higher than the average BMI are {:,.2f}".format(1+(high_bmi-low_bmi)/high_bmi),"times higher than people with average or below average BMI")    


    
    st.markdown("#### Region")
    region_data=df.groupby('region')['charges'].mean()
    st.bar_chart(region_data)
    
    southWestMean=df[df['region']=='southwest']['charges'].mean()
    southEastMean=df[df['region']=='southeast']['charges'].mean()
    northWestMean=df[df['region']=='northwest']['charges'].mean()
    northEastMean=df[df['region']=='northeast']['charges'].mean()
    countryMean=df['charges'].mean()
    charges={southWestMean:'Southwest',southEastMean:'Southeast',northWestMean:'Northwest',northEastMean:'Northeast'}

    charge_list=charges.keys()

    charge_max=max(charges.keys())

    highestRegion=max(charge_list)  

    st.write(" All averages are for Indian regions",
        "\n * National average: ₹{:,.2f}  ".format(countryMean),
        "\n * Southeast average: ₹{:,.2f}".format(charge_max),
        "\n * Northeast average: ₹{:,.2f}".format(northEastMean),
        "\n * Northwest average: ₹{:,.2f}".format(northWestMean),
        "\n * Southwest average: ₹{:,.2f}".format(southWestMean),
        "\n * Region with the highest average fees: ",charges.get(highestRegion),
        )
    
    st.markdown("#### Children")
    customers_with_children=(df[df['children']>0]['charges'].mean()/df[df['children']==0]['charges'].mean()-1)*100
    
    children=df.groupby('children')['charges'].mean()
    st.bar_chart(children)
    st.write("* Average charges for customers without children: ₹{:,.2f}".format(df[df['children']==0]['charges'].mean()),
      "\n * Average charges for customers with children: ₹{:,.2f}".format(df[df['children']>0]['charges'].mean()),
      "\n * Difference in average charges for customers with children: +{:.2f}%".format(customers_with_children))

if __name__ == '__main__':
    main()
