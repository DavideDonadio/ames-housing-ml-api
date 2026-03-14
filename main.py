from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import joblib


app = Flask(__name__)

def train_and_save_model():

    data_path = "house-prices-advanced-regression-techniques/"
    train_data = pd.read_csv(data_path + "train.csv")

    ########################################################
    # HANDLING OUTLIERS, NULL VALUES AND CATEGORICAL DATA ##
    ########################################################

    outliers = train_data.loc[(train_data["SalePrice"] < 200000) & (train_data["GrLivArea"] > 4300), ["GrLivArea", "SalePrice"]]
    train_data = train_data.drop(outliers.index)

    categorical_cols_with_nan = train_data.select_dtypes(include='str').columns
    categorical_cols_with_nan = [col for col in categorical_cols_with_nan if train_data[col].isnull().any()]
    print(categorical_cols_with_nan)

    numerical_cols_with_nan = train_data.select_dtypes(include='number').columns
    numerical_cols_with_nan = [col for col in numerical_cols_with_nan if train_data[col].isnull().any()]
    print(numerical_cols_with_nan)

    for cat in categorical_cols_with_nan:
        train_data[cat] = train_data[cat].fillna(value="None")
        
    for cat in numerical_cols_with_nan:
        if cat == 'GarageYrBlt':
            train_data[cat] = train_data[cat].fillna(0)
        else: 
            train_data[cat] = train_data[cat].fillna(train_data[cat].mean())
            
    train_data = pd.get_dummies(train_data, drop_first=True)

    ########################################################
    ###### FEATURE ENGINEERING AND DATA NORMALIZATION ######
    ########################################################

    x_train = train_data.drop("SalePrice", axis=1)
    x_train['TotalSF'] = x_train['TotalBsmtSF'] + x_train['1stFlrSF'] + x_train['2ndFlrSF']
    x_train = x_train.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)
    
    scaler = StandardScaler()
    numeric_cols = x_train.select_dtypes(include='number').columns
    x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])

    y_train = np.log1p(train_data["SalePrice"])

    ########################################################
    ########## MODEL SELECTION AND EVALUATION ##############
    ########################################################

    model = Ridge(alpha=10.0)
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    print(f"Ridge CV RMSE Scores: {rmse_scores}")
    print(f"Average RMSE: {rmse_scores.mean():.4f}")

    final_model = Ridge(alpha=10.0)
    final_model.fit(x_train, y_train)
    
    os.makedirs('models', exist_ok=True)

    joblib.dump(final_model, 'models/ridge_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(x_train.columns), 'models/model_columns.pkl')

try:
    model = joblib.load('models/ridge_model.pkl') 
    scaler = joblib.load('models/scaler.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    
except FileNotFoundError:
    print("ML Artifacts not found. Starting initial training run...")
    train_and_save_model()
    model = joblib.load('models/ridge_model.pkl') 
    scaler = joblib.load('models/scaler.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    print("Training complete, API is ready.")


@app.route("/")
def hello_world():
    return render_template("home.html") 

@app.route("/predict", methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        df = pd.DataFrame(json_data, index=[0])
                
        if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
            df = df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)
            
        df = pd.get_dummies(df)
        df = df.reindex(columns=model_columns, fill_value=0)
        numeric_cols = scaler.feature_names_in_
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        log_prediction = model.predict(df)
        price = np.expm1(log_prediction[0])
        
        return jsonify({"estimated_price": round(float(price), 2), "status": "Success"})

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/pipeline")
def exercise_pipeline():
    return render_template("exercise_pipeline.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5041, debug=True)