import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("https://raw.githubusercontent.com/avinashkranjan/Loan-Prediction-ML-Project/main/train.csv")

# Preprocessing
df = df.dropna()
df = df.replace({'Loan_Status': {'Y': 1, 'N': 0}})
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)
