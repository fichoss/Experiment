import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
import numpy as np

def preprocess_telco(input_path="/content/drive/MyDrive/Eksperimen_SML_Arfizan-Rabbani/telco_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                     output_dir="/content/drive/MyDrive/Eksperimen_SML_Arfizan-Rabbani/preprocessing/telco_preprocessing"):
    # Load data
    df = pd.read_csv(input_path)

    # Drop kolom yang tidak berguna
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Handle missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target Churn
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Pisahkan fitur dan target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Identifikasi kolom numerik & kategorikal
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Transformasi data
    X_processed = preprocessor.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gabungkan X dan y kembali
    train = np.hstack([X_train.toarray(), y_train.values.reshape(-1, 1)])
    test = np.hstack([X_test.toarray(), y_test.values.reshape(-1, 1)])

    # Buat folder output
    os.makedirs(output_dir, exist_ok=True)

    # Simpan hasil preprocessing
    pd.DataFrame(train).to_csv(f"{output_dir}/train.csv", index=False)
    pd.DataFrame(test).to_csv(f"{output_dir}/test.csv", index=False)




if __name__ == "__main__":
    preprocess_telco()
