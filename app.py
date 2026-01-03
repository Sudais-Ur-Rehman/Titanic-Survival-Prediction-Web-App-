import streamlit as st
import pickle
import numpy as np

# Load trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("Titanic Survival Prediction")

# User input
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", 0, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 32.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Prediction
if st.button("Predict"):
    test_input = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked], dtype=object).reshape(1, 7)
    prediction = pipe.predict(test_input)
    st.success(f"Survival Prediction: {'Survived' if prediction[0]==1 else 'Did not survive'}")