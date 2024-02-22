import streamlit as st

# File upload widget
uploaded_file = st.file_uploader("Choose a file")

# Display the uploaded file
if uploaded_file is not None:
    st.write("File Details:")
    st.write(uploaded_file.name)
    st.write(uploaded_file.type)
    st.write(uploaded_file.size)
