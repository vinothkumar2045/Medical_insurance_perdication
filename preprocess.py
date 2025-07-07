import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path="C:/Users/cpvin/Downloads/medical_insurance.csv"):
    df = pd.read_csv(path)

    # Encode categorical features
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])

    # Optional: Add BMI classification (for EDA only, not for training)
    df['bmi_class'] = pd.cut(df['bmi'],
                             bins=[0, 18.5, 24.9, 29.9, 100],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    df = df.dropna()

    # Drop bmi_class before returning for model training
    df = df.drop(columns=['bmi_class'])

    return df