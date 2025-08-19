import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path=r"C:\Users\cpvin\OneDrive\Documents\Guvi_mini_projects\Medical Insurance Cost Prediction\medical_insurance.csv"):
    """Load dataset and preprocess features + target"""

    df = pd.read_csv(file_path)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Split features & target
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
