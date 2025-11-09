# 🚀 Customer Churn Prediction Dashboard

**AI-Powered Insights & Retention Actions**

A fully interactive Streamlit web app that predicts and analyzes customer churn using **PyCaret AutoML**, **XGBoost**, and **SHAP explainability**.
Deployed live on [DigitalOcean App Platform](https://app.raselmia.live) 🌐

---

## 🧠 Overview

This dashboard enables businesses to:

* Upload customer data (`CSV`)
* Automatically train and optimize machine learning models using **PyCaret AutoML**
* Generate churn predictions instantly
* Visualize explainability insights via **SHAP**
* Understand key drivers of churn and retention

It’s built for data-driven customer retention and actionable decision support.

---

## ⚙️ Tech Stack

| Category            | Tools                     |
| ------------------- | ------------------------- |
| **Frontend**        | Streamlit                 |
| **ML Framework**    | PyCaret (3.3.0)           |
| **Optimization**    | Optuna                    |
| **Explainability**  | SHAP                      |
| **Deployment**      | DigitalOcean App Platform |
| **Language**        | Python 3.10+              |
| **Version Control** | Git & GitHub              |

---

## 🌟 Key Features

✅ **Upload any CSV dataset**
Easily upload and preview your customer dataset directly in the app.

✅ **AutoML for churn prediction**
PyCaret automatically builds and tunes multiple models (XGBoost, LightGBM, CatBoost, etc.) using **Optuna** for hyperparameter optimization.

✅ **Interactive metrics**
Displays dataset summary (rows, columns, missing values, data types) and model KPIs.

✅ **Explainability with SHAP**
Visual breakdown of key features contributing to churn risk.

✅ **AI API Integration (OpenAI)**
Integrates OpenAI’s API to generate retention suggestions and explain model insights in plain English.

✅ **Professional UI**
Clean dark-themed Streamlit interface optimized for deployment.

---

## 📂 Repository Structure

```
customer-churn-dashboard/
│
├── .streamlit/
│   └── config.toml                # Streamlit server configuration
│
├── app.py                         # Main dashboard application
├── Procfile.txt                   # Deployment command for DigitalOcean
├── requirements.txt               # Dependencies
├── .python-version                # Python version specification
├── README.md                      # Project documentation (you’re here)
│
├── models/                        # (Optional) Pre-trained models
│   ├── automl_best_model.pkl
│   └── best_xgboost_model.json
│
└── assets/                        # (Optional) Images, banners, or icons
    └── banner.png
```

---

## ⚡ Installation (Local Setup)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mraselm/customer-churn-dashboard.git
cd customer-churn-dashboard
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## ☁️ Deployment

This app is deployed on **DigitalOcean App Platform**.
To deploy your own version:

1. Push your code to GitHub
2. Connect the repository to DigitalOcean
3. Configure the `Procfile`, `requirements.txt`, and environment variables
4. Expose port `8080` in `.streamlit/config.toml`

### Example `Procfile`

```
web: streamlit run app.py --server.port=8080
```

### Example `.streamlit/config.toml`

```
[server]
headless = true
port = 8080
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
```

---

## 🔐 Environment Variables

| Key              | Description                                |
| ---------------- | ------------------------------------------ |
| `OPENAI_API_KEY` | Required for AI-powered retention insights |

Add this securely in your DigitalOcean App Settings → *Environment Variables*.

---

## 📊 Model Explainability

The dashboard integrates **SHAP (SHapley Additive exPlanations)** to help users understand:

* Which features contribute most to churn
* Individual customer churn reasoning
* Global feature importance trends

This improves transparency and trust in the AutoML predictions.

---

## 🧹 Dependencies

Core libraries (defined in `requirements.txt`):

```
streamlit
pycaret==3.3.0
optuna==<your_local_version>
shap
xgboost
catboost
lightgbm
pandas
numpy
scikit-learn
```

---

## 💡 Future Improvements

* Add persistent model storage with cloud database (PostgreSQL / S3)
* Integrate customer segmentation and retention strategy generation
* Multi-user authentication and dashboard access control

---

## 👨‍💻 Author

**Rasel Mia**
🌍 Aarhus, Denmark
🎓 MSc Business Intelligence, Aarhus University
💼 [LinkedIn](https://linkedin.com/in/mraselm)

---

## 🏁 Live Demo

🔗 [Visit the live app](https://app.raselmia.live)

---

## 📝 License

This project is licensed under the **MIT License** — feel free to use and modify it for your own projects.

---

### 🌈 Credits

Built with [Streamlit](https://streamlit.io/), [PyCaret](https://pycaret.org/), and [DigitalOcean](https://www.digitalocean.com/).
