# ✈️ Airline Passenger Satisfaction Prediction

### Explainable Machine Learning for Airline Customer Experience

This project predicts whether an airline passenger is **satisfied or dissatisfied** and analyzes the service factors associated with passenger experience.

> **Purpose:** I created this project to demonstrate a complete Data Science workflow—from EDA and feature engineering to model comparison, explainability, segmentation and business recommendations.

## 🎯 Business Questions

- Which services influence satisfaction most?
- How do delays affect dissatisfaction?
- Which passenger segments have different satisfaction patterns?
- Can ML reliably identify dissatisfied passengers?

## 🔄 Workflow

```text
Data → Cleaning → EDA → Feature Engineering → Model Comparison
     → Hyperparameter Tuning → Evaluation → Explainability
     → Segmentation → Business Recommendations
```

## 🤖 Models

- Logistic Regression
- Random Forest
- XGBoost

The repository's current notebook/README reports **XGBoost Enhanced** as the strongest model. Re-run the notebook to reproduce current metrics rather than treating historical metrics as permanent.

## 📊 Analysis

Includes:

- Service-rating analysis
- Passenger/class/travel segmentation
- Delay impact analysis
- Feature importance
- SHAP/model explainability
- ROC-AUC, F1 and confusion-matrix evaluation

## 💡 Business Insights

The analysis focuses on digital boarding, Wi-Fi, travel type, class, entertainment, comfort and delay management as actionable experience areas.

## 🚀 Run

```bash
git clone https://github.com/Piyu242005/Airline-Passenger-Satisfaction-Prediction.git
cd Airline-Passenger-Satisfaction-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```

Open the main Jupyter notebook and run the cells sequentially.

## 📁 Structure

```text
Airline satisfaction ML.ipynb   # Main analysis
train.csv                       # Training data
test.csv                        # Test data
*.pkl                           # Model/processing artifacts
README.md
```

## ⚠️ Limitations

- Historical airline data does not prove causal relationships.
- Dataset performance may not generalize to another airline or time period.
- A production version should expose the model through an API and monitor drift.

## 🗺️ Roadmap

- [ ] FastAPI prediction service
- [ ] Interactive Streamlit dashboard
- [ ] Reproducible training pipeline
- [ ] Model monitoring
- [ ] Calibration and threshold optimization

## 👨‍💻 Author

**Piyush Ramteke** — Data Scientist | AI/ML Engineer
