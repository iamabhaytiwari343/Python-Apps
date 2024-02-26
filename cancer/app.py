# iris_dashboard.py
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit App
st.title("Iris Flower Dataset Dashboard")

# Sidebar with user input
st.sidebar.header("User Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', X['sepal length (cm)'].min(), X['sepal length (cm)'].max(), X['sepal length (cm)'].mean())
    sepal_width = st.sidebar.slider('Sepal width', X['sepal width (cm)'].min(), X['sepal width (cm)'].max(), X['sepal width (cm)'].mean())
    petal_length = st.sidebar.slider('Petal length', X['petal length (cm)'].min(), X['petal length (cm)'].max(), X['petal length (cm)'].mean())
    petal_width = st.sidebar.slider('Petal width', X['petal width (cm)'].min(), X['petal width (cm)'].max(), X['petal width (cm)'].mean())
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df_user_input = user_input_features()

# Display the user input
st.subheader('User Input:')
st.write(df_user_input)

# Make predictions
prediction = clf.predict(df_user_input)

# Display the prediction
st.subheader('Prediction:')
st.write(iris.target_names[prediction])

# Display the accuracy of the model
st.subheader('Model Accuracy:')
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'The accuracy of the model on the test set is: {accuracy:.2f}')

