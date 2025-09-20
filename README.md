#🚢 Titanic Survival Predictor
##A web app that predicts whether a passenger would have survived the Titanic disaster based on their details. 
Built using Python, Streamlit, and CatBoost.

##Deployed App
Check out the live app here: https://titanic-survival-ml.streamlit.app/

###Features
Predict survival based on passenger details:
Passenger Class (Pclass)
Sex
Age
Number of Siblings/Spouses aboard (SibSp)
Number of Parents/Children aboard (Parch)
Fare
Port of Embarkation (Embarked)
Encodes categorical features for model prediction.
Simple and interactive Streamlit interface.

###Technologies & Libraries Used
Python – Core programming language
Streamlit – Web app interface
Pandas – Data handling
NumPy – Numerical computations
Matplotlib & Seaborn – Data visualization
CatBoost – Gradient boosting ML model
Joblib – Model saving and loading

###Model Details
Trained on the classic Titanic dataset.
Used CatBoost Classifier for prediction.
Features encoded and transformed for best model performance.
Log transformation applied to Fare as part of preprocessing.
