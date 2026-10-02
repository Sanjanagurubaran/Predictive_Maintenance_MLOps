# Predictive Maintenance Framework for Industrial Equipment Failure Prevention Using MLOps

## 📌 Project Overview

The Predictive Maintenance Framework is a machine learning-based system designed to predict potential industrial equipment failures before they occur. It uses the CatBoost classification algorithm to analyze equipment sensor data and identify possible machine failures.

The project integrates machine learning, explainable AI (SHAP), model deployment, and an interactive dashboard to support data-driven maintenance decisions and reduce unexpected equipment downtime.

## 🎯 Objectives

- Predict potential industrial equipment failures using machine learning.
- Implement CatBoost for accurate failure classification.
- Perform data preprocessing and feature engineering.
- Explain model predictions using SHAP (SHapley Additive exPlanations).
- Develop an interactive dashboard for real-time or input-based predictions.
- Establish an MLOps-oriented workflow for model training, storage, and deployment.

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | CatBoost, Scikit-learn |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Application | Python dashboard (app.py) |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Dataset | AI4I 2020 Predictive Maintenance Dataset |

## 🏗️ System Architecture

The project follows a structured predictive maintenance pipeline:

1. **Data Collection:** Load industrial equipment sensor data.
2. **Data Preprocessing:** Handle missing values, encode categorical variables, and prepare features.
3. **Feature Engineering:** Select and transform relevant equipment parameters.
4. **Model Training:** Train a CatBoost classifier using the prepared dataset.
5. **Model Evaluation:** Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
6. **Explainable AI:** Apply SHAP to understand feature importance and individual predictions.
7. **Model Deployment:** Save the trained model and associated feature information.
8. **Dashboard:** Provide an interface for equipment failure predictions and model insights.

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains simulated industrial equipment operating data, including:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type
- Machine failure indicators
- Failure mode indicators

The dataset is used to train and evaluate the predictive maintenance model.

## 🤖 Machine Learning Model

**CatBoost Classifier**

CatBoost is a gradient boosting algorithm used for classification. It is suitable for structured datasets and supports categorical features.

The model predicts whether industrial equipment is likely to experience failure based on the input sensor measurements.

### Model Evaluation

The project reports a CatBoost test accuracy of **98.35%** in its experimental results.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- SHAP Feature Importance

*Actual performance depends on the dataset, preprocessing, train-test split, and evaluation configuration.*

## 🔍 Explainable AI (SHAP)

SHAP is integrated to improve model interpretability by identifying how individual features contribute to predictions.

Key capabilities:
- Global feature importance
- Individual prediction explanations
- Positive and negative feature contributions
- Visualization of influential equipment parameters

## 📁 Project Structure

```text
Predictive_Maintenance_MLOps/
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│   ├── feature_columns.json
│   ├── predictive_maintenance_catboost.cbm
│   └── predictive_model.pkl
│
├── notebooks/
│   ├── main.ipynb
│   └── CatBoost training and experiment files
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sanjanaguraban/Predictive_Maintenance_MLOps.git
```

### 2. Navigate to the project directory

```bash
cd Predictive_Maintenance_MLOps
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

If the dashboard uses Streamlit, launch it with:

```bash
streamlit run app.py
```

## 📈 Key Features

- CatBoost-based equipment failure prediction
- Industrial sensor data preprocessing
- Model performance evaluation
- SHAP-based explainability
- Saved machine learning models
- Interactive prediction dashboard
- Git-based version control
- MLOps-oriented project organization

## 🚀 Future Enhancements

- Integrate live IoT sensor data.
- Implement automated model retraining.
- Add model monitoring and data drift detection.
- Deploy the application to a cloud platform.
- Integrate automated CI/CD pipelines.
- Implement maintenance alerts and predictive scheduling.

## 👩‍💻 Author

**S Sireya**

B.Tech – Artificial Intelligence and Data Science  
Chennai Institute of Technology, Chennai

## 📜 License

This project is developed for academic and educational purposes. Add an appropriate open-source license if you intend to distribute or reuse the code.

## ⭐ Acknowledgements

- AI4I 2020 Predictive Maintenance Dataset
- CatBoost documentation
- SHAP documentation
- Scikit-learn documentation
