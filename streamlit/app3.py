import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample data
df = sns.load_dataset("iris")

# Display a table
st.write("Sample Data:")
st.dataframe(df)

# Create a scatter plot
st.write("Scatter Plot:")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=ax)
st.pyplot(fig)
