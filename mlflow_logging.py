import mlflow
import mlflow.sklearn
from train_models import evaluate_model
from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def log_model_with_mlflow():
    df = load_and_preprocess()
    X = df.drop(['charges'], axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_params({"n_estimators": 100, "model_type": "RandomForest"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_model")

if __name__ == "__main__":
    log_model_with_mlflow()