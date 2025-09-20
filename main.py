import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model once
# model = joblib.load('Titanic Dataset Analysis - Classification/TITANIC-APP/TITANIC-APP/catboost_model.pkl')
model = joblib.load('catboost_model.pkl')


# Streamlit app
st.title("🚢 Titanic Survival Predictor ")
st.write("""
Fill in the passenger details below and predict if they would have survived the Titanic disaster.
""")

#Input feilds
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
SibSp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", min_value=0, step=1)
Parch = st.number_input("Number of Parents/Children aboard (Parch)", min_value=0, step=1)
Fare = st.number_input("Passenger Fare", min_value=0.0, step=0.1)
Embarked = st.selectbox("Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

# Encoding the categorical inputs
sex_encoded = 1 if Sex == "Male" else 0
embarked_encoded = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}[Embarked]


# Fare_log = np.log1p(fare_input + 1)
#  #apply lof transformations as we trained using log
fare_log = np.log(Fare + 1)

# Predict button
if st.button("Predict Survival"):
    features = np.array([[Pclass, sex_encoded, Age, SibSp, Parch, Fare, embarked_encoded]])
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.success("🎉 The passenger would have **SURVIVED!**")
    else:
        st.error("😢 The passenger would **NOT have survived.**")

st.sidebar.subheader("🔍 Feature Importance")  #starting as a side heading

feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
importances = model.get_feature_importance()
imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis', ax=ax)
st.sidebar.pyplot(fig)

if st.sidebar.button("🔄 Reset"):
    st.experimental_rerun() #clear/reset button!
