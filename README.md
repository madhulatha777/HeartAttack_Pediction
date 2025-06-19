# 💓 Heart Attack Risk Prediction

This is a Streamlit-based web application that predicts the risk of a heart attack based on a user's health and lifestyle information. The model was trained using an XGBoost classifier, and the application incorporates preprocessing for categorical and numerical features.

---

## 🚀 Features

- Simple and interactive web interface built with **Streamlit**
- Accepts key medical and lifestyle indicators as inputs
- Uses a **trained XGBoost model** for binary classification (high or low risk)
- Preprocessing handled via **pickle-based transformer**
- Blood pressure input handled as `systolic/diastolic` (e.g., `120/80`)
- Shows **predicted probability** of heart attack risk

---

## 🧠 Model Details

- **Model**: XGBoost Classifier
- **Preprocessing**: 
  - `Sex` and `Diet` are treated as categorical (one-hot encoded)
  - All other features are treated as numerical
- **Trained on**: Cleaned and balanced dataset with binary labels (`0`: Low Risk, `1`: High Risk)

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Frontend**: Streamlit  
- **Models**: XGBoost (final selected models)  
- **Libraries**: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `pickle`  
- **Data Preprocessing**: StandardScaler

---

## 🖥️ Running the App Locally

1. **Clone the repository:**

```bash
git clone https://github.com/madhulatha777/HeartAttack_Pediction.git
cd HeartAttack_Prediction
```

2. **Install the Dependencies**

```bash
pip install -r requirements.txt
```

3. **Launch the Streamlit app**

```bash
streamlit run app.py
```

---

## 🔍 App Preview

Below is a screenshot of the Heart Attack Prediction Streamlit web app:

![Streamlit App Screenshot](...)

![Streamlit App Screenshot](...)

---

## 🤝 Contributions
Feel free to fork this repository and contribute. Open a pull request for suggestions, bug fixes, or improvements!

---

## 🙏 Acknowledgments
- UCI Machine Learning Repository for the Heart Attack Prediction Dataset

- Streamlit for the rapid prototyping tool

- Scikit-learn and XGBoost for powerful modeling frameworks

---

## ✉️ Contact
For any queries or collaborations, reach out via [LinkedIn](https://www.linkedin.com/in/madhulatha-seerapu-8a269b325/) or email: madhuseerapu1@gmail.com