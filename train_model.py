import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Accuracy
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc*100:.2f}%")

# Save everything
joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/encoder.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

# Save dataset for visualization
df.to_csv("model/processed_data.csv", index=False)

print("Model trained and saved ✅")