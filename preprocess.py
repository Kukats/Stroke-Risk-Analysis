import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)

    # Handle missing values
    if "bmi" in df.columns:
        df['bmi'].fillna(df['bmi'].mean(), inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Features/target split
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns
