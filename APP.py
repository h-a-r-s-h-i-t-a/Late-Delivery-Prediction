import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from xgboost import DMatrix

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


st.title('Enhancing Supply Chain Efficiency: Predictive Modeling for Timely Deliveries')

st.sidebar.header("User Input Parameters")

shipping_mode_Standard_Class = st.sidebar.selectbox("Shipping Mode Standard Class", ["Yes","No"])
shipping_time_Morning = st.sidebar.selectbox("Shipping Time Morning", ["Yes", "No"])
shipping_week_day = st.sidebar.selectbox("Shipping Week Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
order_week_day = st.sidebar.selectbox("Order Week Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
order_time = st.sidebar.selectbox("Order Time", ["Morning", "Afternoon", "Night"])

input_data = pd.DataFrame({
    "Shipping Mode": [shipping_mode_Standard_Class],
    "Shipping Time": [shipping_time_Morning],
    "Shipping Week Day": [shipping_week_day],
    "Order Week Day": [order_week_day],
    "Order Time": [order_time],
})
input_data_encoded = pd.get_dummies(input_data, drop_first=True)

st.write("User Input:", input_data)

prediction = model.predict(input_data)
st.write(f"Prediction: {prediction}")

fig = px.bar(x=['Late', 'On Time'], y=[50, 50], labels={'x': 'Delivery Status', 'y': 'Count'})
st.plotly_chart(fig)

st.header("Project Summary")
st.write("""
    This project focuses on optimizing supply chain efficiency by predicting late deliveries using a classification model.
    The model achieves an accuracy of 97%, enabling proactive measures to reduce delays in the supply chain.
""")