import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained files
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
features = joblib.load("model/features.pkl")

# ✅ FIX: Load dataset safely
df = pd.read_csv("data/Training.csv")

# Remove unwanted column if exists
if 'Unnamed: 133' in df.columns:
    df = df.drop(columns=['Unnamed: 133'])

st.title("🧠 AI Healthcare Diagnosis System")
st.write("Select your symptoms and get predicted diseases")

# Remove symptom_count from UI
symptom_options = [f for f in features if f != "symptom_count"]

# Symptom selection
symptoms = st.multiselect("Select Symptoms:", symptom_options)

# Convert input
input_data = np.zeros(len(features))

for symptom in symptoms:
    if symptom in features:
        idx = features.index(symptom)
        input_data[idx] = 1

# Handle symptom_count
symptom_count = np.sum(input_data)

if "symptom_count" in features:
    idx = features.index("symptom_count")
    input_data[idx] = symptom_count

# ------------------ PREDICTION ------------------
if st.button("Predict Disease"):

    probs = model.predict_proba([input_data])[0]

    top3_idx = np.argsort(probs)[-3:][::-1]
    diseases = encoder.inverse_transform(top3_idx)
    confidences = probs[top3_idx]

    st.subheader("Top 3 Predictions")

    for i in range(3):
        st.write(f"{diseases[i]} → {confidences[i]*100:.2f}%")

    # ------------------ BAR CHART ------------------
    st.subheader("📊 Prediction Confidence (Bar Chart)")
    fig1, ax1 = plt.subplots()
    ax1.bar(diseases, confidences)
    ax1.set_ylabel("Confidence")
    st.pyplot(fig1)

    # ------------------ PIE CHART ------------------
    st.subheader("🥧 Confidence Distribution (Pie Chart)")
    fig2, ax2 = plt.subplots()
    ax2.pie(confidences, labels=diseases, autopct='%1.1f%%')
    st.pyplot(fig2)

    # ------------------ SEVERITY ------------------
    if symptom_count > 10:
        severity = "Severe"
    elif symptom_count > 5:
        severity = "Moderate"
    else:
        severity = "Mild"

    st.subheader("Severity Level")
    st.write(severity)

    # ------------------ RECOMMENDATION ------------------
    st.subheader("Recommendation")

    if severity == "Severe":
        st.warning("⚠️ Consult a specialist doctor immediately")
    elif severity == "Moderate":
        st.info("💊 Monitor symptoms and consult doctor if needed")
    else:
        st.success("✅ Maintain healthy lifestyle")

# ------------------ DATA VISUALIZATION ------------------

st.subheader("📈 Dataset Insights")

# Disease distribution
st.write("### Disease Distribution")
fig3, ax3 = plt.subplots()
df['prognosis'].value_counts().head(10).plot(kind='bar', ax=ax3)
st.pyplot(fig3)

# Symptom frequency
st.write("### Symptom Frequency")
symptom_counts = df.drop('prognosis', axis=1).sum().sort_values(ascending=False).head(10)

fig4, ax4 = plt.subplots()
symptom_counts.plot(kind='bar', ax=ax4)
st.pyplot(fig4)