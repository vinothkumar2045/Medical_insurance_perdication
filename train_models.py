import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from preprocess import load_and_preprocess  # Import your preprocessing function

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

# Main function to train models and save the best one
def train_and_save_best_model():
    df = load_and_preprocess()

    # ✅ Drop non-numeric 'bmi_class' column if it exists
    if 'bmi_class' in df.columns:
        df = df.drop(columns=['bmi_class'])

    X = df.drop(['charges'], axis=1)
    y = df['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define candidate models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }

    best_score = -1
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        print(f"{name} performance: {metrics}")

        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = model
            best_name = name

    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"✅ Best Model Saved: {best_name}")

if __name__ == "__main__":
    train_and_save_best_model()