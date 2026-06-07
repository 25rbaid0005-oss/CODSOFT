import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸")

# Title
st.title("🌸 Iris Flower Classification")
st.write("Predict the species of an Iris flower using Machine Learning.")

# Load Dataset
df = pd.read_csv("archive (3)/IRIS.csv")

# Display Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features and Target
X = df.drop("species", axis=1)
y = df["species"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Sidebar Inputs
st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    float(df["sepal_length"].min()),
    float(df["sepal_length"].max()),
    float(df["sepal_length"].mean())
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    float(df["sepal_width"].min()),
    float(df["sepal_width"].max()),
    float(df["sepal_width"].mean())
)

petal_length = st.sidebar.slider(
    "Petal Length",
    float(df["petal_length"].min()),
    float(df["petal_length"].max()),
    float(df["petal_length"].mean())
)

petal_width = st.sidebar.slider(
    "Petal Width",
    float(df["petal_width"].min()),
    float(df["petal_width"].max()),
    float(df["petal_width"].mean())
)

# Input Data
input_data = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

st.subheader("Input Values")
st.write(input_data)

# Prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)

# Result
st.subheader("Prediction")

if prediction == "Iris-setosa":
    st.success("🌼 Iris Setosa")
elif prediction == "Iris-versicolor":
    st.success("🌸 Iris Versicolor")
else:
    st.success("🌺 Iris Virginica")

# Probability
st.subheader("Prediction Probabilities")

prob_df = pd.DataFrame(
    probability,
    columns=model.classes_
)

st.dataframe(prob_df)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.info(f"{accuracy*100:.2f}%")