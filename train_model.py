import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("data/Training.csv")

# Clean
if 'Unnamed: 133' in df.columns:
    df = df.drop(columns=['Unnamed: 133'])

df = df.drop_duplicates()

# Encode target
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])

# Feature engineering
df['symptom_count'] = df.drop('prognosis', axis=1).sum(axis=1)

# Split
X = df.drop('prognosis', axis=1)
y = df['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/encoder.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

print("Model trained and saved ")