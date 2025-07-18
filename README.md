# 🔍 Customer Churn Prediction Dashboard

An interactive dashboard that predicts whether a customer will churn using machine learning and SHAP explanations. Users can upload customer data and instantly receive churn probability, decision confidence, and recommended actions.

🚀 **Live App**: [https://app.raselmia.live](https://app.raselmia.live)  
🌐 **Portfolio**: [https://raselmia.live](https://raselmia.live)

---

## 📌 What It Does

- ✅ Predicts **customer churn likelihood**
- 📈 Displays **model confidence** and **SHAP explanation plots**
- 👥 Highlights **top features driving the prediction**
- 🎯 Suggests **business actions** based on churn risk
- 📂 Supports **CSV file uploads** for batch predictions

---

## 📂 Example File Format

To ensure compatibility, your input `.csv` should match the format used during training.  
📥 A sample file is included in this repo as `processed_churn_dataset.csv`.

**Required Columns:**
- `CustomerID`
- `Tenure`
- `PreferredLoginDevice`
- `CityTier`
- *(Other features used by the model)*

If the structure does not match, the app will display an error. A sample file is provided inside the repo.

---

## ⚙️ Tech Stack

- Streamlit
- XGBoost
- SHAP
- pandas, numpy, scikit-learn

---

## 🛠️ Run Locally

```bash
git clone https://github.com/mraselm/churn-dashboard.git
cd churn-dashboard
pip install -r requirements.txt
streamlit run app.py
