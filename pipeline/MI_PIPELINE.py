from prefect import flow, task
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

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

    print("Descargando datos desde URL (simulando API)...")

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

    # GUARDAR DATA LIMPIA
    df.to_csv("data_processed.csv", index=False)

    return df

# ----------------------
# SELECT BEST MODEL
# ----------------------
@task
def select_best_model(models):

    best_model_name = min(models, key=lambda x: models[x][1])
    best_model, best_mae = models[best_model_name]

    print(" Mejor modelo:", best_model_name)
    print("MAE:", best_mae)

    return best_model, best_model_name, best_mae


# ----------------------
# TRAIN MODEL
# ----------------------
@task
def train_model(df):

    X = df[["Quantity", "UnitPrice", "Year", "Month"]]
    y = df["TotalPrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("Retail Project")

    with mlflow.start_run():
        mlflow.log_artifact("data_processed.csv")

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_preds)

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_preds = lr_model.predict(X_test)
        lr_mae = mean_absolute_error(y_test, lr_preds)

        # Gradient Boosting
        gb_model = GradientBoostingRegressor()
        gb_model.fit(X_train, y_train)
        gb_preds = gb_model.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_preds)

        # Imprimir resultados
        print("Random Forest MAE:", rf_mae)
        print("Linear Regression MAE:", lr_mae)
        print("Gradient Boosting MAE:", gb_mae)

        # Diccionario de modelos
        models = {
            "RandomForest": (rf_model, rf_mae),
            "LinearRegression": (lr_model, lr_mae),
            "GradientBoosting": (gb_model, gb_mae)
        }

        # Selección del mejor modelo (nueva tarea)
        best_model, best_model_name, best_mae = select_best_model(models)

        # Registro en MLflow
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("mae", best_mae)

        mlflow.sklearn.log_model(best_model, "model")

    return best_model

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