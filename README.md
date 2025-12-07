# ✈️ Airline Passenger Satisfaction Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A machine learning project to predict airline passenger satisfaction based on flight experience data including service quality, seat comfort, delays, and other factors.

## 📋 Project Overview

Airlines want to understand what factors influence passenger satisfaction to improve customer experience and reduce dissatisfaction rates. This project builds a predictive model that:

- Predicts whether a passenger is **Satisfied** or **Dissatisfied**
- Identifies key factors affecting customer satisfaction
- Provides actionable business recommendations

## 📊 Dataset

- **Source:** [Kaggle - Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Training samples:** ~104,000 records
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
10. Insights & Recommendations
```

## 🤖 Models Implemented

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~87% | ~0.85 |
| Random Forest | ~96% | ~0.95 |
| XGBoost | ~96% | ~0.95 |

**Best Model:** Random Forest (with GridSearchCV optimization)

## 🏆 Key Findings

### Top Factors Affecting Satisfaction:
1. **Online Boarding** - Most important predictor
2. **Inflight WiFi Service** - Strong impact on satisfaction
3. **Type of Travel** - Business vs Personal travel
4. **Class** - Business class has higher satisfaction
5. **Inflight Entertainment** - Key service differentiator

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
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
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
- Delay analysis
- Feature importance chart
- ROC curves and Confusion matrices

## 💡 Business Recommendations

1. **Improve Online Services** - Enhance online boarding and WiFi quality
2. **Focus on Business Travelers** - Streamline check-in, priority services
3. **Minimize Delays** - Better communication and compensation
4. **Upgrade Comfort** - Seat comfort and entertainment systems
5. **Class-specific Strategies** - Tailored services for each class

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **Joblib/Pickle** - Model serialization

## 📝 Author

**Piyush Ramteke**

## 📄 License

This project is for educational purposes as part of an internship project.

---

⭐ If you found this project helpful, please give it a star!
