# 🧠 AI Healthcare Diagnosis System

## 📌 Overview

An AI-powered healthcare diagnostic system that predicts possible diseases based on user-selected symptoms.
This project is designed to assist patients with early diagnosis, support doctors in decision-making, and help hospitals optimize triage.

---

## 🚀 Features

* 🔍 Disease prediction based on symptoms
* 📊 Top-3 probable diseases with confidence scores
* ⚠️ Severity classification (Mild / Moderate / Severe)
* 💡 Smart recommendations for next steps
* 🌐 Interactive web app using Streamlit

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (Random Forest Classifier)
* XGBoost (for experimentation)
* Streamlit (UI & Deployment)
* Joblib (model persistence)

---

## ⚙️ How It Works

1. User selects symptoms via the web interface
2. Symptoms are converted into binary feature vectors
3. Model predicts probabilities for diseases
4. Top 3 diseases are displayed with confidence scores
5. Severity is calculated based on symptom count
6. Recommendations are provided accordingly

---

## 📂 Project Structure

```
healthcare-ai/
│
├── data/
│   ├── Training.csv
│   ├── Testing.csv
│
├── model/
│   ├── model.pkl
│   ├── encoder.pkl
│   ├── features.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance

* Algorithm: Random Forest Classifier
* Achieved high accuracy on test data (~90%+)
* Supports multi-disease prediction (Top-3 results)

---

## 🌐 Live Demo

👉 (https://diseasepredictionhealthcare-ai.streamlit.app)

---

## 👥 Team Members

This project was developed collaboratively by a team of 4:

* Saiyan
* Bishnu
* Maggi
* Chetan

---

## ⚠️ Disclaimer

This system is intended for educational and research purposes only.
It should not be used as a substitute for professional medical advice.

---

## 🚀 Future Improvements

* 🤖 Chatbot-based symptom input
* 📊 Advanced data visualization dashboard
* 🧠 Explainable AI (SHAP integration)
* 🏥 Doctor/specialist recommendation system
* 📱 Mobile-friendly UI

---

## ⭐ Contribution

Feel free to fork the repo, raise issues, and contribute!

---

## 📜 License

This project is open-source and available under the MIT License.
