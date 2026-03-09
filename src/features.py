import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# segregate the numeric colums from the
# categorical columns because the treatments are different
NUMERIC_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
]

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]

TARGET = 'Churn'
DROP_COLS = ['customerID']

def load_and_prepare(filepath: str) -> tuple:
    df = pd.read_csv(filepath)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df.drop(columns=DROP_COLS, inplace=True)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == 'Yes').astype(int)

    return X, y

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=
                                       ('onehot', OneHotEncoder(
                                           handle_unknown='ignore',
                                           sparse_output=False
                                       )))
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer,    NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder='drop'
    )

    return preprocessor

def get_train_test_split(X,y, test_size = 0.2, random_state = 42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


if __name__ == '__main__':
    X, y = load_and_prepare('data/churn_raw.csv')
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Churn rate train: {y_train.mean():.1%}")
    print(f"Churn rate test:  {y_test.mean():.1%}")

    preprocessor = build_preprocessor()
    print(f"Preprocessor built successfully")
    print("✓ features.py smoke test passed")
    