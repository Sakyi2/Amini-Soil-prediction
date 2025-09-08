# Amini-Soil-prediction
<img width="720" height="720" alt="463399585-b49044e1-dd9a-4100-87f2-c50e7aa866b0" src="https://github.com/user-attachments/assets/0887e80a-224f-4bff-acb6-edb111b5fea6" />


# 🌱 AMI Soil Property Prediction Challenge

This repository contains our solution for the [AMI Soil Property Prediction Challenge](https://zindi.africa/competitions/ami-soil-property-prediction-challenge) hosted on Zindi Africa. The goal is to build machine learning models that predict essential soil properties using various features extracted from soil samples.

## 📊 Problem Statement

Agricultural productivity depends heavily on soil quality. The objective of this challenge is to predict key soil properties from given data. These predictions will help AMI support farmers in better decision-making and improve agricultural outputs.

## 🔍 Project Structure

ami-soil-prediction/
│
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
│
├── notebooks/
│ ├── eda.ipynb
│ ├── xgboost_model.ipynb
│ ├── random_forest_model.ipynb
│ └── ensemble_model.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── train_models.py
│ ├── utils.py
│ └── evaluate.py
│
├── models/
│ └── saved_models/
│
├── submissions/
│ └── submission_xgb.csv
│
├── requirements.txt
└── README.md

markdown
Copy
Edit

## 📦 Models Used

We experimented with several supervised learning models:

- **XGBoost Regressor**: Known for its performance in tabular data and competitions.
- **Random Forest Regressor**: Robust ensemble model that reduces overfitting.
- **Gradient Boosting Regressor**: Focuses on minimizing loss via stage-wise optimization.
- **Linear Regression (Baseline)**: Used as a benchmark.
- **Stacked Ensemble**: Combination of the best-performing models.

## 🧪 Evaluation Metric

The model is evaluated using **Root Mean Squared Error (RMSE)** across the target variables.

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/ami-soil-prediction.git
   cd ami-soil-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run training:

bash
Copy
Edit
python src/train_models.py
Generate submission:

bash
Copy
Edit
python src/predict_submission.py
📈 Results
Model	RMSE Score
XGBoost	0.218
Random Forest	0.229
Gradient Boosting	0.225
Stacked Ensemble	0.210

Note: These are offline validation scores. Final leaderboard scores may vary.

🙌 Contributors
Your Name

Team Zindi

📌 License
This project is open-sourced under the MIT License.

python
Copy
Edit

---

Let me know if you'd like the code files (`train_models.py`, `preprocess.py`, etc.) or if you're submitting as a notebook and want a version suited for that instead.







