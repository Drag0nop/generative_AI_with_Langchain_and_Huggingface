import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title("Streamlit Widgets Example")

name = st.text_input("Enter some text:")
if name:
    st.write(f"You entered: {name}")