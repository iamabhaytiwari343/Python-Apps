import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load Iris dataset
@st.cache_data
def load_data():
    iris_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                          header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    return iris_df

iris_df = load_data()

# Sidebar - Model Training
st.sidebar.title("Model Training")

# Select features and target variable
features = st.sidebar.multiselect("Select features", iris_df.columns[:-1], default=["sepal_length", "sepal_width"])
target_variable = st.sidebar.selectbox("Select target variable", iris_df.columns[-1])

# Encoding categorical target variable
if target_variable in iris_df.columns:
    iris_df[target_variable] = le.fit_transform(iris_df[target_variable])
else:
    st.sidebar.error(f"Selected target variable '{target_variable}' not found in the dataset.")
    st.stop()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    iris_df[features], iris_df[target_variable], test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sidebar - Model Evaluation
st.sidebar.title("Model Evaluation")

# Display model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")

# Display classification report
st.sidebar.write("Classification Report:")
st.sidebar.text(classification_report(y_test, y_pred))

# Main content
st.title("Iris Dataset Dashboard")

# Display dataset
st.write("## Iris Dataset")
st.dataframe(iris_df)

# Scatter plot
st.write("## Scatter Plot")
fig, ax = plt.subplots()
sns.scatterplot(data=iris_df, x=features[0], y=features[1], hue=target_variable, ax=ax)
st.pyplot(fig)

# Model Predictions
st.write("## Model Predictions")

# User input for prediction
sepal_length = st.slider("Select Sepal Length", float(iris_df[features[0]].min()), float(iris_df[features[0]].max()), float(iris_df[features[0]].mean()))
sepal_width = st.slider("Select Sepal Width", float(iris_df[features[1]].min()), float(iris_df[features[1]].max()), float(iris_df[features[1]].mean()))

# Make prediction
input_data = pd.DataFrame({"sepal_length": [sepal_length], "sepal_width": [sepal_width]})
prediction = le.inverse_transform(model.predict(input_data))[0]

# Display prediction
st.write(f"Predicted Class: {prediction}")
