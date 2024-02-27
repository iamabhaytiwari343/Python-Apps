# cancer_dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Standardize the data for pair plot
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pairplot = pd.DataFrame(X_scaled, columns=cancer.feature_names)
X_pairplot['target'] = y

# Streamlit App
st.title("Breast Cancer Dataset Dashboard")

# Sidebar with user input
st.sidebar.header("User Input Features")

def user_input_features():
    features = {}
    for feature in cancer.feature_names:
        features[feature] = st.sidebar.slider(feature, X[feature].min(), X[feature].max(), X[feature].mean())
    data = pd.DataFrame(features, index=[0])
    return data

df_user_input = user_input_features()

# Display the user input
st.subheader('User Input:')
st.write(df_user_input)

# Make predictions
prediction = clf.predict(df_user_input)

# Display the prediction
st.subheader('Prediction:')
st.write("Malignant" if prediction[0] == 1 else "Benign")

# Display the accuracy of the model
st.subheader('Model Accuracy:')
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'The accuracy of the model on the test set is: {accuracy:.2f}')

# Display classification report
st.subheader('Classification Report:')
st.text(classification_report(y_test, y_pred))

# Pair plot for data exploration
st.subheader('Pair Plot:')
fig, ax = plt.subplots()
sns.pairplot(X_pairplot, hue='target', height=2)
st.pyplot(fig)

# Confusion Matrix
st.subheader('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, linewidths=.5)
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(fig)
