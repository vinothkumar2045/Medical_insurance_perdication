import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df):
    df = df.copy()
    df.dropna(inplace=True)

    cat_cols = ['sex', 'smoker', 'region']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # ✅ FIXED HERE
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    df_numeric = df.drop(columns=cat_cols, errors='ignore').reset_index(drop=True)
    df_final = pd.concat([df_numeric, encoded_df], axis=1)

    if 'charges' in df_final.columns:
        X = df_final.drop('charges', axis=1)
        y = df_final['charges']
    else:
        X = df_final
        y = None

    return X, y, encoder



def train_models(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        print(f"{name} RMSE: {rmse:.2f}")
        if rmse < best_score:
            best_score = rmse
            best_model = model

    return best_model
