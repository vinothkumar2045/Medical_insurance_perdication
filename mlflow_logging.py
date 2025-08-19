import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_and_preprocess

def run_experiment():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(model, "linear_regression_model")

        print(f"Logged metrics: MSE={mse:.2f}, R²={r2:.2f}")

if __name__ == "__main__":
    run_experiment()
