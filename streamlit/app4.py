import streamlit as st

# Text input widget
name = st.text_input("Enter your name", "John Doe")

# Display the entered name
st.write(f"Hello, {name}!")
