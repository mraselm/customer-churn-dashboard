# Customer Churn Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-ff4b4b)](https://streamlit.io/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3-orange)](https://pycaret.org/)

A Streamlit dashboard for customer churn prediction, model validation, explainability, and retention planning.

The app lets you upload a customer CSV, train churn models with PyCaret AutoML, inspect individual churn risk, review SHAP explanations, segment customers with RFM analysis, and generate retention recommendations. It also includes safeguards for common churn-modeling problems such as data leakage, ID-column leakage, overly optimistic metrics, and unlabeled datasets.

## Features

- CSV upload with automatic schema handling and demo-data support.
- Supervised churn prediction using PyCaret AutoML.
- Optional unsupervised churn-label generation when no target column exists.
- Professional validation engine for leakage detection and realistic performance checks.
- Monitoring agent for training errors, fallbacks, and recovery messages.
- RFM customer value profiling for recency, frequency, and monetary segmentation.
- SHAP explanations for customer-level churn drivers.
- Counterfactual and prescriptive retention recommendations.
- Optional OpenAI-powered insight generation and promotional messaging.
- Deployment-ready Streamlit configuration through `Procfile.txt`.

## Repository Structure

```text
.
|-- app.py                              # Main Streamlit application
|-- validation_engine.py                # Leakage detection and model validation
|-- monitoring_agent.py                 # Training monitoring and recovery helpers
|-- unsupervised_churn.py               # Churn label inference for unlabeled data
|-- universal_rfm.py                    # RFM feature engineering and segmentation
|-- requirements.txt                    # Full local dependency set
|-- requirements_deploy.txt             # Deployment-focused dependency set
|-- Procfile.txt                        # Streamlit startup command for cloud platforms
|-- QUICK_START.md                      # Short usage guide
|-- PROFESSIONAL_VALIDATION_GUIDE.md    # Validation details
|-- MONITORING_AGENT_GUIDE.md           # Monitoring agent details
|-- RFM_INTEGRATION_GUIDE.md            # RFM integration details
|-- test_*.py                           # Component test scripts
`-- thesis_results/                     # Generated analysis outputs
```

## Requirements

- Python 3.10 or newer
- pip
- A virtual environment is recommended

The full dependency list includes machine-learning packages such as PyCaret, XGBoost, LightGBM, CatBoost, SHAP, DiCE, scikit-learn, pandas, NumPy, Plotly, and Streamlit. Installation can take a few minutes.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/mraselm/customer-churn-dashboard.git
cd customer-churn-dashboard
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Configuration

OpenAI features are optional. The dashboard still works without an API key.

To enable AI-generated recommendations, create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

For Streamlit Cloud, add the same value to Streamlit secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Do not commit `.env` or any secret files.

## Using The Dashboard

1. Upload a CSV file from the sidebar, or use the included demo dataset if available.
2. Select the churn target column and customer ID column if prompted.
3. If no churn column exists, enable unsupervised mode to infer churn-like labels.
4. Click `Run AutoML`.
5. Review validation messages, model diagnostics, and the AutoML leaderboard.
6. Select an individual customer to inspect churn probability, RFM profile, SHAP drivers, and suggested retention actions.

## Data Expectations

The app expects a tabular CSV file with a header row.

Recommended columns:

- A binary churn target such as `Churn`, `Exited`, `Cancelled`, `Left`, or similar.
- A customer identifier such as `CustomerID`, `customer_id`, `AccountID`, or similar.
- Behavioral, usage, billing, tenure, support, product, or transaction fields.

Supported target values include common binary formats such as:

- Churned: `1`, `yes`, `true`, `churn`, `churned`, `left`, `exited`
- Retained: `0`, `no`, `false`, `stay`, `stayed`, `retained`, `active`

ID-like columns are excluded from model training to reduce leakage risk.

## Validation And Leakage Protection

The dashboard includes a validation layer built around `ChurnValidationEngine`.

It checks for:

- Features that are suspiciously correlated with the target.
- ID-like fields that should not be used for training.
- RFM-derived fields that should be shown for analysis but excluded from model training.
- Unrealistic performance such as near-perfect AUC.
- Train/test gaps that may indicate overfitting.

Validation results are shown in the dashboard during training and in the model overview section after training.

## Testing

Run the component tests individually:

```bash
python test_validation.py
python test_monitoring_agent.py
python test_unsupervised_churn.py
python test_reliability.py
python test_universal.py
```

If you use pytest, you can also run:

```bash
pytest
```

Some tests may take longer because they exercise machine-learning and data-processing paths.

## Deployment

The included `Procfile.txt` starts Streamlit with a platform-provided port:

```text
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
```

Typical deployment steps:

1. Push the repository to GitHub.
2. Create a Streamlit, Heroku, DigitalOcean, or similar app from the repository.
3. Use `app.py` as the entry point.
4. Install dependencies from `requirements.txt` or `requirements_deploy.txt`.
5. Add `OPENAI_API_KEY` as an environment variable if AI recommendations are needed.
6. Allocate enough memory for PyCaret and model training. At least 2 GB is recommended; 4 GB is safer for larger datasets.

## Documentation

Additional guides are included in the repository:

- [QUICK_START.md](QUICK_START.md)
- [PROFESSIONAL_VALIDATION_GUIDE.md](PROFESSIONAL_VALIDATION_GUIDE.md)
- [MONITORING_AGENT_GUIDE.md](MONITORING_AGENT_GUIDE.md)
- [RFM_INTEGRATION_GUIDE.md](RFM_INTEGRATION_GUIDE.md)
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- [DASHBOARD_CHANGES_SUMMARY.md](DASHBOARD_CHANGES_SUMMARY.md)

## Security Notes

- Keep API keys in environment variables or Streamlit secrets.
- Do not commit `.env`, model artifacts containing sensitive data, or customer data exports.
- Review uploaded data carefully before using OpenAI-powered text generation.
- For production deployments, add authentication if the dashboard will handle private customer data.

## Author

Rasel Mia  
MSc Business Intelligence, Aarhus University

## License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2026 Rasel Mia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
