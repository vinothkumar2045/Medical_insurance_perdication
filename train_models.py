import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_and_preprocess

def train_and_save_model():
    """Train model and save best one"""

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # Train simple Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:\nMSE = {mse:.2f}, R² = {r2:.2f}")

    # Save model + scaler
    joblib.dump(model, "best_model.pkl")
    joblib.dump(scaler, "caler.pkl")

    print("✅ Model and scaler saved in 'models/'")

if __name__ == "__main__":
    train_and_save_model()
