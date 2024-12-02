import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


st.title("Restaurant Tips Prediction App")
st.subheader("This App predicts the tip value that customer pay")

st.sidebar.header('User Input Parameter')

def user_input_features():
	total_bill = st.sidebar.slider('Total bill', 3.07,50.81,12.07)
	size = st.sidebar.slider('Size',1 , 6 , 2)
	day = st.sidebar.selectbox('Day of Week', ['Thur','Fri','Sat','Sun'])
	smoker = st.sidebar.selectbox('Smoker',['Yes','No'])
	time = st.sidebar.selectbox('time',['Dinner','Lunch'])
	sex = st.sidebar.selectbox('Sex',['Male','Female'])


	data = {
        'total_bill': total_bill,
        'sex': 1 if sex == "Male" else 0,
        'smoker': 1 if smoker == "Yes" else 0,
        'day': {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}[day],
        'time': 1 if time == "Dinner" else 0,
        'size': size
    }
   
	features = pd.DataFrame(data, index=[0])
	return features


df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

tips = sns.load_dataset('tips')

tips.replace({ 'sex': {'Male':0 , 'Female':1} , 'smoker' : {'No': 0 , 'Yes': 1}} ,inplace=True)

label_encoder = LabelEncoder()
tips['day'] = label_encoder.fit_transform(tips['day'])  # Thur=0, Fri=1, Sat=2, Sun=3
tips['time'] = label_encoder.fit_transform(tips['time'])  # Lunch=0, Dinner=1

X = tips.drop('tip',axis=1)
Y = tips['tip']

regressor = LinearRegression()
regressor.fit(X,Y)

predictions = regressor.predict(df)


st.subheader('Prediction tip:')
st.write(predictions)
