[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rileybarka/Lab-5-Airbnb-Model-Evaluation/blob/main/notebooks/Lab5.ipynb)

# Lab 5: Logistic Regression Model Selection & Evaluation

This lab continues the machine learning lifecycle, focusing on the **evaluation** and **selection** of logistic regression models using the Airbnb NYC listings dataset. The emphasis is on comparing baseline vs. tuned models, understanding trade-offs via precision-recall and ROC analysis, selecting informative features, and saving the trained model for future use.

---

## Dataset Used

| Dataset | Description |
|---------|-------------|
| **Airbnb Listings NYC (Dec 2021)** | Modified Airbnb dataset used to define classification labels, engineer features, and evaluate logistic regression models. |

---

## Objectives

- Define the classification problem (label and features)  
- Train a baseline logistic regression with default hyperparameters  
- Perform grid search to tune the regularization parameter (`C`)  
- Retrain using the optimal hyperparameter  
- Plot and compare precision-recall curves for both models  
- Plot ROC curves and compute AUC for both models  
- Apply feature selection to improve input quality  
- Persist the final trained model for later inference  

---

## Methodology

1. Load and prepare the Airbnb dataset (feature engineering, label definition, train-test split)  
2. Train baseline logistic regression using scikit-learn defaults  
3. Run grid search over `C` (inverse regularization strength) to find the best value  
4. Retrain with optimal `C`  
5. Evaluate both models:
   - Precision-Recall curves  
   - ROC curves and AUC  
6. Perform feature selection to reduce dimensionality or remove noisy inputs  
7. Save/serialize the best model (e.g., with `joblib` or `pickle`) for reuse  

---

## Key Outputs (to populate after running)

- Baseline vs tuned model performance (precision, recall, AUC)  
- Precision-recall comparison plot  
- ROC curves with AUC annotations  
- Selected feature subset  
- Serialized model artifact (e.g., `best_logistic_model.pkl`)  

---

## Setup Instructions

### Option 1: Open in Google Colab  
Click the badge above.

### Option 2: Run Locally

```bash
git clone https://github.com/rileybarka/Lab-5-Airbnb-Model-Evaluation.git
cd Lab-5-Airbnb-Model-Evaluation/notebooks
jupyter notebook Lab5.ipynb
