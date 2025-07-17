import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title("Data Analysis App")

df = pd.DataFrame({
    'first_column': [1, 2, 3, 4, 5],
    'second_column': [10, 20, 30, 40, 50]
})

st.write("Here is a simple DataFrame:")
st.write(df)

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)

# Set the title of the app
st.title("Streamlit Widgets Example")

name = st.text_input("Enter some text:")
if name:
    st.write(f"You entered: {name}")