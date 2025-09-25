# 📊 Customer Churn Prediction

This project predicts customer churn for a telecom company using machine learning.  
It demonstrates the full data science workflow — from **EDA and preprocessing** to **model training, evaluation, and explainability**.  
Developed in **Google Colab** with Python.

---

## 🚀 Project Workflow

1. **Data Loading & Cleaning**
   - Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - Handled missing values in `TotalCharges`
   - Encoded categorical features (e.g., Contract type, Payment method)
   - Balanced target labels using **SMOTE**

2. **Exploratory Data Analysis (EDA)**
   - Visualized churn distribution
   - Examined churn by contract type, tenure, and charges
   - Identified key drivers of churn: short tenure, month-to-month contracts, high monthly charges

3. **Feature Engineering & Preprocessing**
   - Standardized numerical features with `StandardScaler`
   - Created new grouped features (e.g., tenure groups)
   - Prepared train/test splits with stratification

4. **Model Training**
   - Baseline: Logistic Regression
   - Advanced: Random Forest, XGBoost
   - Hyperparameter tuning via GridSearchCV (optional extension)

5. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Visuals: Confusion Matrix, ROC Curve
   - Best performing model: **XGBoost**

6. **Explainability**
   - Used **SHAP values** to interpret model predictions
   - Key insights: 
     - Customers on month-to-month contracts are much more likely to churn
     - Lower tenure strongly correlates with churn
     - High monthly charges increase churn probability

---

## 📈 Results

- **Best Model:** XGBoost  
- **Performance:** ROC-AUC ~ 0.85, with balanced Precision and Recall  
- **Key Drivers of Churn:**
  - Contract type (month-to-month contracts churn the most)
  - Short tenure (new customers more likely to churn)
  - High monthly charges

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn, Imbalanced-learn
- XGBoost
- Matplotlib, Seaborn
- SHAP (explainability)
- Google Colab (development environment)

---

## 📂 Repository Structure

