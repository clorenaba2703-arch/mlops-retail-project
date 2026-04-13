from prefect import flow, task
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import mlflow
import mlflow.sklearn

# ----------------------
# LOAD DATA
# ----------------------
import requests
from io import BytesIO

@task
def load_data():

    print("🌐 Descargando datos desde URL (simulando API)...")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

    response = requests.get(url)

    df = pd.read_excel(BytesIO(response.content))

    print("Datos cargados:", df.shape)

    return df

# ----------------------
# PREPROCESS
# ----------------------
@task
def preprocess_data(df):

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month

    return df

# ----------------------
# TRAIN MODEL
# ----------------------
@task
def train_model(df):

    X = df[["Quantity", "UnitPrice", "Year", "Month"]]
    y = df["TotalPrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlflow.set_experiment("Retail Project")  # 👈 aquí va esto

    with mlflow.start_run():

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")

    print("MAE:", mae)

    return model

# ----------------------
# FLOW
# ----------------------
@flow
def retail_pipeline():

    df = load_data()
    df_clean = preprocess_data(df)
    model = train_model(df_clean)

    print("Pipeline completo ejecutado")

if __name__ == "__main__":
    retail_pipeline()