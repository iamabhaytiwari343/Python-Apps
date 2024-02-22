import streamlit as st

# Slider widget
age = st.slider("Select your age", 0, 100, 25)

# Button widget
if st.button("Say Hello"):
    st.write(f"Hello! You are {age} years old.")
