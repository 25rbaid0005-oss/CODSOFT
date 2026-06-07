import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------
# PAGE SETTINGS
# -----------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger survived the Titanic disaster.")

# -----------------------
# LOAD DATASET
# -----------------------
@st.cache_data
def load_data():
    # Change train.csv if your file has a different name
    return pd.read_csv(r"titanic.csv\train.csv")

df = load_data()

# -----------------------
# DATA CLEANING
# -----------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical columns
sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

df["Sex"] = sex_encoder.fit_transform(df["Sex"])
df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

# -----------------------
# MODEL TRAINING
# -----------------------
X = df[[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.success(f"Model Accuracy: {accuracy:.2%}")

# -----------------------
# USER INPUTS
# -----------------------
st.header("Enter Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.slider(
    "Age",
    1,
    80,
    25
)

sibsp = st.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked_choice = st.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

# Convert values
sex = 1 if gender == "male" else 0
embarked = embarked_encoder.transform([embarked_choice])[0]

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("🎉 Passenger Would Survive")
    else:
        st.error("❌ Passenger Would Not Survive")

    st.write(f"Survival Probability: {probability:.2%}")

# -----------------------
# SHOW DATA
# -----------------------
with st.expander("View Dataset"):
    st.dataframe(df.head())
