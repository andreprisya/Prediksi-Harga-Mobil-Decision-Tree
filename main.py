import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
import csv


st.header("Prediksi Harga Mobil Dengan Metode Decision Tree")
# st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("Car.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.bin")

st.subheader("A. Dataset")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)
st.write("OR")
st.write("Input Data Column")
input_horsepower = st.text_input('Horsepower : ')
input_price = st.text_input('Price ($) : ')

if st.button('Input Data'):
    df_input = pd.DataFrame(
      {
        "horsepower":[input_horsepower],
        "price":[input_price]})
    data.append(df_input)
    st.write(data)

st.write("")
st.write("Final Dataset")


st.write("")
st.write("")
st.subheader("B. Masukkan Data Prediksi!")

# left_column, right_column = st.columns(2)
# with left_column:
#     inp_species = st.radio(
#         'Name of the fish:',
#         np.unique(data['Species']))

# input_Length1 = st.slider('Vertical length(cm)', 0.0, max(data["Length1"]), 1.0)
# input_Length2 = st.slider('Diagonal length(cm)', 0.0, max(data["Length2"]), 1.0)
# input_Length3 = st.slider('Cross length(cm)', 0.0, max(data["Length3"]), 1.0)
# input_Height = st.slider('Height(cm)', 0.0, max(data["Height"]), 1.0)
# input_Width = st.slider('Diagonal width(cm)', 0.0, max(data["Width"]), 1.0)

input_prediksi_horsepower = st.text_input('Masukkan Horsepower : ')

if st.button('Make Prediction'):
    # input_species = encoder.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [int(input_prediksi_horsepower)], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Harga Mobil Adalah : {np.squeeze(prediction, -1)} $")

    # st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
    # st.write(f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)")



