# Ghana Rent Price Predictor

A Machine Learning project that predicts estimated residential property listing prices in Ghana using property characteristics and location data.

## Project Objectives

- Analyse Ghanaian residential property listing data.
- Identify factors influencing property prices.
- Develop regression models for price prediction.
- Compare models using appropriate evaluation metrics.
- Build a reusable prediction system.
- Deploy the model for public use.

## Dataset

The project uses the Ghana House Rental Dataset from Kaggle.

**Dataset source:** [Ghana House Rental Dataset](https://www.kaggle.com/datasets/epigos/ghana-house-rental-dataset/)

The dataset contains 17,890 property listings and 17 original columns, including:

- Property type and category
- Number of bedrooms and bathrooms
- Floor area
- Location, region and locality
- Property condition
- Furnishing status
- Parking availability
- Amenities
- Listed price

The dataset is used to investigate relationships between property characteristics, location and listed prices.

## Machine Learning Techniques

The project explores regression techniques, including:

- Linear Regression
- Random Forest Regressor

Additional models planned for future experimentation:

- Decision Tree Regressor
- XGBoost Regressor

## Model Evaluation

The Random Forest Regressor was selected as the current baseline model.

### Test Set Results

| Metric | Result |
|---|---:|
| MAE | GHS 2,978.01 |
| RMSE | GHS 5,734.58 |
| R² | 0.7175 |

### 5-Fold Cross-Validation

| Metric | Result |
|---|---:|
| Mean MAE | GHS 2,890.90 |
| MAE Standard Deviation | GHS 71.94 |
| Mean R² | 0.7250 |

Cross-validation indicates relatively consistent model performance across the five folds.

### Model Limitations

- Prediction errors increase for higher-priced properties.
- Expensive listings tend to be underestimated.
- The dataset contains a strong concentration of listings from Greater Accra.
- Some floor-area values may be inconsistent.
- The payment period represented by the listed prices requires further verification.

Predictions should therefore be treated as estimates rather than guaranteed market prices.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost (planned)
- Jupyter Notebook
- Joblib
- Git and GitHub

## Project Structure

```text
ghana-rent-predictor/
├── .github/
│   └── workflows/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Current Progress

- [x] Project setup and environment configuration
- [x] Data exploration
- [x] Data cleaning and preprocessing
- [x] Feature engineering
- [x] Model training
- [x] Model comparison
- [x] Model evaluation
- [x] Residual analysis
- [x] Feature importance analysis
- [x] 5-fold cross-validation
- [ ] Reusable prediction module
- [ ] Automated prediction tests
- [ ] Django REST API
- [ ] React frontend
- [ ] Deployment

## Future Development

- Improve prediction performance, particularly for expensive properties.
- Investigate additional location-based features.
- Experiment with additional regression algorithms.
- Build a reusable prediction module.
- Develop a Django REST API.
- Build a React frontend.
- Deploy the application for public use.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ghana-rent-predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Disclaimer

Predictions are estimates based on historical property listing data. They are not official property valuations, professional appraisals or guaranteed market prices.