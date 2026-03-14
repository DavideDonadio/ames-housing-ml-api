# Ames Housing Price Predictor API 🏠

A production-ready, dockerized Machine Learning service that estimates residential house prices in Ames, Iowa, using an optimized Ridge Regressor.

## Project Overview
This project transforms the classic Ames Housing dataset into a deployable REST API. It handles the entire machine learning lifecycle, from raw data preprocessing to a containerized prediction service.

### Key Features
* **Model:** Ridge Regression (Linear model with L2 Regularization).
* **Performance:** Achieved an Average RMSE of **0.1157** (RMSLE) via 5-fold Cross-Validation.
* **Pipeline:** Automated data cleaning, outlier removal, feature engineering (TotalSF), and input scaling.
* **Deployment:** Flask API served within a Docker container using `uv` for modern, reproducible dependency management.

## Stack
* **Language:** Python 3.10
* **ML Libraries:** Scikit-Learn (Ridge), Pandas, Numpy
* **API Framework:** Flask
* **Environment/DevOps:** Docker, uv, Joblib

## Data & Engineering Insights
- **Target Transformation:** To handle the right-skewed nature of house prices, a log1p transformation was applied, ensuring a more normal distribution for the linear model.
- **Feature Engineering:** Combined basement and floor square footage into a single TotalSF feature to reduce multicollinearity and improve model stability.
- **Preprocessing:** Implemented a robust pipeline involving median imputation for missing values and StandardScaling for numerical features.

## How to Run

### 1. Build the Image
Navigate to the project root and run:
```bash
docker build -t ames-api .
```

### 2. Start the Service
Map the container's port to your local machine:
```bash
docker run -p 5041:5041 ames-api
```

### 3. Get a Prediction
Open a new terminal and send a POST request with house features:
```bash
curl -X POST http://localhost:5041/predict \
  -H "Content-Type: application/json" \
  -d '{"OverallQual":8,"GrLivArea":2000,"GarageCars":2,"TotalBsmtSF":1000,"1stFlrSF":1000,"2ndFlrSF":800}'
```
