from flask import Flask, render_template, request, jsonify, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import joblib


app = Flask(__name__)
data_path = "house-prices-advanced-regression-techniques/"

train_data = pd.read_csv(data_path + "train.csv")
test_data = pd.read_csv(data_path + "test.csv")

########################################################
################# DATASET INSPECTION ###################
########################################################

rows, columns = train_data.shape
print(f"Working with: {rows} rows and {columns} columns\n")

print("Dataset Non-null count and types: \n")
train_data.info()

print("Dataset Count, Mean, STD, MIN, MAX, Percentiles: \n")
print(train_data.describe())

print("Target variable histogram: \n")
#sns.histplot(train_data["SalePrice"])

print("Correlation between targert variable and all other columns: \n")
train_corr = train_data.corr(numeric_only=True)
top_features = train_corr["SalePrice"].sort_values(ascending=False).head(10)
print(top_features)

corr_top = train_data[top_features.index[:10]].corr()
#sns.heatmap(corr_top, annot=True)

########################################################
# HANDLING OUTLIERS, NULL VALUES AND CATEGORICAL DATA ##
########################################################

# Two clear outliers are visible from here, we filter them out.
#sns.scatterplot(x=train_data["GrLivArea"], y=train_data["SalePrice"]) 
outliers = train_data.loc[(train_data["SalePrice"] < 200000) & (train_data["GrLivArea"] > 4300), ["GrLivArea", "SalePrice"]]
print(outliers["GrLivArea"])

# We find those outliers and we remove them from the training data
train_data = train_data.drop(outliers.index)

# We show columns with NaN values and their percentange
missingOverallQual = train_data.isnull().sum().sort_values(ascending=False)
missingOverallQual = missingOverallQual[missingOverallQual > 0].sort_values(ascending=False)
missingPercent = (train_data.isnull().sum() / len(train_data) * 100).sort_values(ascending=False)
print(missingOverallQual)
print(missingPercent)

# We find and substitute NaN values intelligently, ex. PoolQC NaN means no pool so we insert --> "None"
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
################# DATA NORMALIZATION ###################
########################################################

x_train = train_data.drop("SalePrice", axis=1)

# We found 3 columns that can be represented as one, i decided to drop them and keep the new one
x_train['TotalSF'] = x_train['TotalBsmtSF'] + x_train['1stFlrSF'] + x_train['2ndFlrSF']
x_train = x_train.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)
print(x_train.describe())   
   
scaler = StandardScaler()
numeric_cols = x_train.select_dtypes(include='number').columns
x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])

y_train = np.log1p(train_data["SalePrice"])

sns.histplot(y_train)

########################################################
########## MODEL SELECTION AND FINE-TUNING #############
########################################################

model = Ridge(alpha=10.0)
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)

print(f"Ridge CV RMSE Scores: {rmse_scores}")
print(f"Average RMSE: {rmse_scores.mean():.4f}")

#Train the final model on ALL of your clean training data
final_model = Ridge(alpha=10.0)
final_model.fit(x_train, y_train)

#Save the Model, the data Scaler and the column names
joblib.dump(final_model, 'ridge_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(x_train.columns), 'model_columns.pkl')

model = joblib.load('ridge_model.pkl') 
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')


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
        \
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