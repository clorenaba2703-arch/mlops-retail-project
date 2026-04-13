print("🔥 SI VES ESTO, YA FUNCIONA")

from prefect import flow, task
import pandas as pd

@task
def load_data():
    df = pd.read_excel("data/raw/online_retail.xlsx")
    print("Datos cargados:", df.shape)
    return df

@flow
def retail_pipeline():
    df = load_data()

if __name__ == "__main__":
    retail_pipeline()