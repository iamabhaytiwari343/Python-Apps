import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the digits dataset from scikit-learn
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a Streamlit dashboard
st.title("Digits Dataset Dashboard")

# Display dataset summary
st.write("## Digits Dataset Summary")
st.write("Number of samples:", X.shape[0])
st.write("Number of features:", X.shape[1])
st.write("Number of classes:", len(set(y)))

# Display model accuracy
st.write("## Model Accuracy")
st.write("Accuracy:", accuracy)

# Allow users to make predictions
st.write("## Make Predictions")
user_data = []

for i in range(X.shape[1]):
    feature_value = st.slider(f"Pixel Value for Feature {i}", float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
    user_data.append(feature_value)

user_data = [user_data]  # Convert to 2D array for prediction

prediction = clf.predict(user_data)

st.write("## Prediction")
st.write("Predicted digit:", prediction[0])
