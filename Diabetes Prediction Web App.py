# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:00:42 2024

@author: Prateek Yadav
"""

import numpy as np 
import pickle
import streamlit as st

# Loading the Saved model
loaded_model = pickle.load(open('C:/Users/Prateek Yadav/Desktop/PIMA Diabetes Prediction Using SVM/trained_model.sav', 'rb'))


# Create a function for prediction

def diabetes_prediction(input_data):
    

    # Changing the input data to the numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is non diabetic'
    else:
        return 'The person is diabetic'
    
def main():
    
    # Title of the web app
    st.title('Diabetes Prediction Web Application')
    
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Blood Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Blood Insulin Level')
    BMI = st.text_input('Body Mass Index (BMI) Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')
    
   
    # Code for prediction
    diagnosis = ''
    
    # creating a button for prediction 
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction ,Age])
    
    st.success(diagnosis)


if __name__ == '__main__':
    main()