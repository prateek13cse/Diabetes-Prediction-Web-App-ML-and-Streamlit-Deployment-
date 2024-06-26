import numpy as np
import pickle

# Loading the Saved model
loaded_model = pickle.load(open('C:/Users/Prateek Yadav/Desktop/PIMA Diabetes Prediction Using SVM with Streamlit Deployment/trained_model.sav', 'rb'))

input_data = (7,83,78,26,71,29.3,0.767,36)

# Changing the input data to the numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person is non diabetic')
else:
    print('The person is diabetic')
    
    
    
    
    


