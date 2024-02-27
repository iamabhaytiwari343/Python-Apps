# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a simple DataFrame for demonstration
data = pd.DataFrame({
    'x': np.random.rand(100),
    'y': np.random.rand(100)
})

# Streamlit app
st.title('Simple Scatter Plot Dashboard')

# Display the DataFrame
st.subheader('DataFrame:')
st.dataframe(data)

# Scatter plot
st.subheader('Scatter Plot:')
fig, ax = plt.subplots()
ax.scatter(data['x'], data['y'])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
st.pyplot(fig)

# Add a slider to adjust the number of data points
num_points = st.slider('Number of data points:', min_value=10, max_value=1000, value=100)

# Update the plot based on the selected number of data points
updated_data = pd.DataFrame({
    'x': np.random.rand(num_points),
    'y': np.random.rand(num_points)
})

st.subheader(f'Scatter Plot with {num_points} data points:')
fig, ax = plt.subplots()
ax.scatter(updated_data['x'], updated_data['y'])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
st.pyplot(fig)
