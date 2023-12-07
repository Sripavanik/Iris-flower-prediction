import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('trained_model.joblib')

# Define a function to make predictions
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit App
st.title('Iris Flower Prediction App')

# Add sliders for user input
sepal_length = st.slider('Sepal Length', float(4.3), float(7.9), float(5.4))
sepal_width = st.slider('Sepal Width', float(2.0), float(4.4), float(3.4))
petal_length = st.slider('Petal Length', float(1.0), float(6.9), float(1.3))
petal_width = st.slider('Petal Width', float(0.1), float(2.5), float(0.2))

# Make a prediction based on user input
prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)

# Display the predicted species
species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
predicted_species = species_mapping[prediction]

st.subheader('Predicted Iris Species:')
st.write(predicted_species)
