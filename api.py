from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow.sklearn

# 🔹 1. CREAR APP PRIMERO
app = FastAPI()

# 🔹 2. CARGAR MODELO
import joblib

model = joblib.load("model.pkl")
FEATURES = list(model.feature_names_in_)

# 🔹 4. VALIDACIÓN
class InputData(BaseModel):
    Quantity: int = Field(gt=0)
    UnitPrice: float = Field(gt=0)
    Month: int = Field(ge=1, le=12)
    Year: int = Field(ge=2010, le=2025)

# 🔹 5. ENDPOINTS
@app.get("/")
def home():
    return {"message": "API de modelo retail activa 🚀"}

@app.post("/predict")
def predict(data: InputData):

    try:
        df = pd.DataFrame([data.model_dump()])

        df = df.loc[:, FEATURES]

        prediction = model.predict(df)[0]

        return {
            "prediction": round(float(prediction), 2),
            "currency": "GBP",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))