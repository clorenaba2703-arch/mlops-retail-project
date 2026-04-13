# Prefect pipeline for retail ML project
from prefect import flow, task
import pandas as pd

@task
def load_data():
    df = pd.read_excel("data/raw/online_retail.xlsx")
    return df

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

@flow
def retail_pipeline():

    df = load_data()
    df_clean = preprocess_data(df)

    print("Datos finales:", df_clean.shape)

if __name__ == "__main__":
    retail_pipeline()
    @task
def train_model(df):

    X = df.drop("TotalPrice", axis=1)
    y = df["TotalPrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
