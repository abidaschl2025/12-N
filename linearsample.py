import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Load data from CSV
df=pd.read_csv("student_scores.csv")
#Split data into features (x) and target (y)
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values
#Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, state = 42)
#Train the model
model = LinearRegression()
model.fit(X_train, y_train)
#Streamlit user interface
st.title("Exam Score Predictor")
st.write("Enter hours studied to predict the exam score.")
#User input
hours = st.number_input("Hours Studied:", min_value=0.0, step=0.1)
#Predict button
if st.button("Predict Score:"):
  predicted_score = model.predict([[hours]])[0]
  st.success(f"Predicted Score: (predicted_score: .2f)")
#Show sample data
st.write("### Sample Training Data")
st.dataframe(df)
