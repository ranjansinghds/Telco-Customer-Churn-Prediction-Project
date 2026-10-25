
# Telco Customer Churn Prediction 

## 📌 Project Overview

This project focuses on predicting customer churn and identifying the factors that influence customers to leave a telecom company.

The project uses **Exploratory Data Analysis, Data Preprocessing, Machine Learning, and Streamlit** to build a customer churn prediction system.

## 🎯 Objectives

- Analyze customer behavior and churn patterns.
- Identify factors associated with customer churn.
- Build and compare classification models.
- Select the best-performing model.
- Predict customers who are likely to churn.
- Provide business recommendations for customer retention.

## 🚀 Streamlit Application

The trained model is integrated into a Streamlit application that allows users to enter customer information and receive a churn prediction with its probability.

### Sample Prediction:
![Churn Prediction](https://github.com/ranjansinghds/Telco-Customer-Churn-Prediction-Project/blob/main/Telco%20Customer%20Churn%20Prediction%20Png/sample%20prediction.png)

### Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔄 Project Workflow

```text
→ Data Cleaning
→ EDA
→ Feature Engineering
→ Preprocessing
→ Model Building
→ Cross-Validation
→ Model Comparison
→ Hyperparameter Tuning
→ Final Evaluation
→ Deployment
```
## 🔍 Methodology

The project follows a structured machine learning workflow to analyze customer behavior and predict customer churn.

### 1. Data Collection
- Used the **Telco Customer Churn dataset** containing customer demographic, service, contract, billing information and so on.
- The target variable is `Churn`, which indicates whether a customer has left the company.

### 2. Data Cleaning
- Checked the dataset structure, data types, missing values, and duplicate records.
- Converted `TotalCharges` from `object` to numeric format.
- Handled missing values created during data type conversion.
- Removed the `customerID` column because it is a unique identifier and does not provide useful predictive information.
- Converted the `Churn` target into binary values:
  - `No → 0`
  - `Yes → 1`

### 3. Exploratory Data Analysis (EDA)
- Analyzed the distribution of customer churn.
- Studied relationships between churn and important features such as:
  - Tenure
  - Monthly Charges
  - Contract Type
  - Internet Service
  - Payment Method
  - Customer Services
- Used visualizations such as count plots, box plots and bar plots analysis to identify important patterns.

Here are some visualizations from the project:

![alt text](https://github.com/ranjansinghds/Telco-Customer-Churn-Prediction-Project/blob/main/Telco%20Customer%20Churn%20Prediction%20Png/pngchurn_distribution)
![alt text](https://github.com/ranjansinghds/Telco-Customer-Churn-Prediction-Project/blob/main/Telco%20Customer%20Churn%20Prediction%20Png/churn_by_contract_type.png)
![alt text](https://github.com/ranjansinghds/Telco-Customer-Churn-Prediction-Project/blob/main/Telco%20Customer%20Churn%20Prediction%20Png/tenure_vs_churn.png)
![alt text](https://github.com/ranjansinghds/Telco-Customer-Churn-Prediction-Project/blob/main/Telco%20Customer%20Churn%20Prediction%20Png/churn_by_internet_service.png)


### 4. Data Preprocessing
- Separated the features (`X`) and target (`y`).
- Split the dataset into training and testing sets using stratified sampling.
- Applied different preprocessing techniques to numerical and categorical features:
  - **Numerical features:** Mean imputation + StandardScaler
  - **Categorical features:** Most-frequent imputation + OneHotEncoder
- Used `ColumnTransformer` and `Pipeline` to organize the preprocessing and modeling workflow. This also helps prevent data leakage during cross-validation.

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 📁 Project Files

```text
├── Telecom_Customer_Churn_Prediction.ipynb
├── Telco_Customer_Churn_Dataset.csv
├── app.py
├── churn_model.pkl
├── requirements.txt
└── Telecom Customer Churn Prediction Png/
```

## 💡 Key Outcome

The project demonstrates how machine learning can be used to identify customers at risk of churn and support **data-driven customer retention strategies**.
