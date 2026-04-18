from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn

app = FastAPI()

#  CARGAR MODELO DESDE MLFLOW
model = mlflow.sklearn.load_model(
    "runs:/07285efcb2464af18bbb6f1f8388c148/model"
)

@app.get("/")
def home():
    return {"message": "API de modelo retail activa 🚀"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return {
        "prediction": round(prediction, 2),
       "description": "Valor total estimado de la transacción en libras esterlinas (GBP)"
    }