# ✈️ Airline Passenger Satisfaction Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A comprehensive machine learning project to predict airline passenger satisfaction based on flight experience data including service quality, seat comfort, delays, and other factors.

## 📋 Project Overview

Airlines want to understand what factors influence passenger satisfaction to improve customer experience and reduce dissatisfaction rates. This project builds a predictive model that:

- Predicts whether a passenger is **Satisfied** or **Dissatisfied**
- Identifies key factors affecting customer satisfaction
- Performs **Customer Segmentation Analysis** for targeted improvements
- Analyzes **Delay Impact** on passenger satisfaction
- Provides actionable business recommendations

## 📊 Dataset

- **Source:** [Kaggle - Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Training samples:** ~104,000 records
- **Test samples:** ~26,000 records
- **Features:** 25 columns including:
  - Passenger demographics (Age, Gender)
  - Flight information (Class, Type of Travel, Flight Distance)
  - Service ratings (Wifi, Seat Comfort, Entertainment, Food, Cleanliness, etc.)
  - Delay information (Departure/Arrival delays)

## 🔧 Project Workflow

```
1. Problem Definition
       ↓
2. Data Understanding
       ↓
3. Data Preprocessing
       ↓
4. Exploratory Data Analysis (EDA)
       ↓
5. Feature Engineering
       ↓
6. Train-Test Split (70-30)
       ↓
7. Model Building
       ↓
8. Model Evaluation
       ↓
9. Hyperparameter Tuning
       ↓
10. Delay Impact Analysis
       ↓
11. Customer Segmentation Analysis
       ↓
12. Key Insights & Recommendations
       ↓
13. Model Deployment (Save Artifacts)
```

## 🤖 Models Implemented

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~87% | ~0.85 | ~0.93 |
| Random Forest | ~96% | ~0.95 | ~0.99 |
| XGBoost Enhanced | **96.32%** | **0.9568** | **0.9949** |

**Best Model:** XGBoost Enhanced (with GridSearchCV optimization)

## 🏆 Key Findings

### Top Factors Affecting Satisfaction:
1. **Online Boarding** - Most important predictor
2. **Inflight WiFi Service** - Strong impact on satisfaction
3. **Type of Travel** - Business vs Personal travel
4. **Class** - Business class has higher satisfaction
5. **Inflight Entertainment** - Key service differentiator

### Delay Impact Analysis:
- **Dissatisfied passengers** experience ~4 minutes more delay on average
- **No Delay:** 54.2% dissatisfaction rate
- **>60 min Delay:** 64.3% dissatisfaction rate
- Delays significantly correlate with increased dissatisfaction

### Customer Segmentation Insights:
| Segment | Satisfaction Rate | Population % |
|---------|------------------|--------------|
| Business Premium | 72.02% | 45.72% |
| Business Economy | 29.62% | 19.50% |
| Standard Economy | 18.43% | 12.81% |
| Leisure Premium | 12.24% | 2.08% |
| Young Leisure | 10.27% | 10.52% |
| Senior Leisure | 9.95% | 9.38% |

### Business Insights:
- 📱 **Digital Services Matter:** Online boarding and WiFi are top predictors
- ✈️ **Business Travelers:** Higher expectations, prioritize efficiency
- ⏱️ **Delays Hurt:** Departure delays significantly impact dissatisfaction
- 🛋️ **Comfort Counts:** Seat comfort and legroom affect experience

## 📁 Project Structure

```
├── Airline satisfaction ML.ipynb    # Main ML notebook
├── train.csv                        # Training dataset
├── test.csv                         # Test dataset
├── optimized_rf_model.pkl           # Saved model
├── label_encoders.pkl               # Encoding artifacts
├── scaler.pkl                       # Scaling artifacts
├── feature_columns.pkl              # Feature list
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

### Run the Project
1. Clone the repository
2. Place `train.csv` and `test.csv` in the project folder
3. Open `Airline satisfaction ML.ipynb` in Jupyter Notebook
4. Run all cells sequentially

## 📈 Visualizations

The notebook includes:
- Satisfaction distribution (bar & pie charts)
- Age distribution analysis
- Satisfaction by Class and Travel Type
- Service ratings distributions
- Correlation heatmap
- **Delay Impact Analysis** charts
- **Customer Segmentation** visualizations
- Feature importance chart
- ROC curves and Confusion matrices

## 💡 Business Recommendations

1. **🌐 Online Services Improvement**
   - Enhance online boarding experience (highest importance)
   - Improve inflight WiFi service quality
   - Simplify ease of online booking

2. **🛋️ In-Flight Comfort**
   - Upgrade seat comfort, especially in Economy class
   - Improve inflight entertainment options
   - Enhance legroom service

3. **⏱️ Delay Management**
   - Minimize departure delays (major dissatisfaction factor)
   - Improve communication during delays
   - Offer compensation for significant delays

4. **👥 Customer Type Focus**
   - Loyal customers have higher expectations - offer premium services
   - Business travelers prioritize efficiency - streamline check-in
   - Personal travelers value entertainment - enhance entertainment options

5. **🎯 Class-Specific Strategies**
   - Economy: Focus on basic comfort and cleanliness
   - Business: Premium services and priority boarding
   - First Class: Personalized experience and exclusive amenities

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **SHAP** - Model interpretability
- **Joblib/Pickle** - Model serialization

## 📝 Author

**Piyush Ramteke**

## 📄 License

This project is for educational purposes as part of an internship project.

---

⭐ If you found this project helpful, please give it a star!
