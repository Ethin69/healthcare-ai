import streamlit as st
import numpy as np
import joblib

# Load trained files
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
features = joblib.load("model/features.pkl")

st.title("🧠 AI Healthcare Diagnosis System")
st.write("Select your symptoms and get predicted diseases")

# Remove 'symptom_count' from UI (IMPORTANT)
symptom_options = [f for f in features if f != "symptom_count"]

# Symptom selection
symptoms = st.multiselect("Select Symptoms:", symptom_options)

# Convert input into model format
input_data = np.zeros(len(features))

for symptom in symptoms:
    if symptom in features:
        idx = features.index(symptom)
        input_data[idx] = 1

# Correct way to handle symptom_count
symptom_count = np.sum(input_data)

if "symptom_count" in features:
    idx = features.index("symptom_count")
    input_data[idx] = symptom_count

# Predict
if st.button("Predict Disease"):
    
    probs = model.predict_proba([input_data])[0]

    top3_idx = np.argsort(probs)[-3:][::-1]
    diseases = encoder.inverse_transform(top3_idx)
    confidences = probs[top3_idx]

    st.subheader("Top 3 Predictions")

    for i in range(3):
        st.write(f"{diseases[i]} → {confidences[i]*100:.2f}%")

    # Severity classification
    if symptom_count > 10:
        severity = "Severe"
    elif symptom_count > 5:
        severity = "Moderate"
    else:
        severity = "Mild"

    st.subheader("Severity Level")
    st.write(severity)

    # Recommendation
    st.subheader("Recommendation")

    if severity == "Severe":
        st.warning("⚠️ Consult a specialist doctor immediately")
    elif severity == "Moderate":
        st.info("💊 Monitor symptoms and consult doctor if needed")
    else:
        st.success("✅ Maintain healthy lifestyle")