# 🛍️ MLOps Retail Project

## 📊 Descripción

Este proyecto implementa un flujo completo de Machine Learning aplicado a datos de ventas retail, incluyendo:

* Análisis Exploratorio de Datos (EDA)
* Limpieza y transformación de datos
* Ingeniería de características
* Entrenamiento de modelo
* Tracking con MLflow
* API de inferencia con FastAPI
* Pipeline automatizado con Prefect

El objetivo es predecir el **valor total de una transacción (TotalPrice)** a partir de variables clave del negocio.

---

## 🎯 Objetivo

Desarrollar un sistema de Machine Learning capaz de:

* Analizar patrones de ventas
* Identificar productos y clientes clave
* Modelar el comportamiento del negocio
* Exponer el modelo mediante una API lista para producción

---

## 📁 Dataset

El dataset corresponde a transacciones de una tienda online e incluye:

* `InvoiceNo`: número de factura
* `StockCode`: código del producto
* `Description`: descripción del producto
* `Quantity`: cantidad
* `UnitPrice`: precio unitario
* `CustomerID`: identificador del cliente
* `Country`: país
* `InvoiceDate`: fecha de transacción

---

## 🧹 Limpieza de datos

Se realizaron las siguientes transformaciones:

* Eliminación de valores nulos en `CustomerID`
* Eliminación de cancelaciones (facturas con "C")
* Eliminación de valores negativos en `Quantity` y `UnitPrice`

📉 El dataset se redujo de **541,909 → 397,884 registros**

---

## ⚙️ Ingeniería de características

Se crearon variables clave para el modelo:

* `TotalPrice = Quantity * UnitPrice`
* `Year`, `Month`, `Day` (variables temporales)

---

## 📈 Análisis Exploratorio (EDA)

Se realizaron visualizaciones para:

* Ventas por mes
* Productos más vendidos
* Ventas por país
* Clientes más valiosos
* Distribución de precios (con y sin outliers)
* Relación entre precio, cantidad e ingresos

---

## 💡 Insights de negocio

* Existe estacionalidad en ventas (picos en fin de año)
* Un subconjunto de productos genera la mayor parte del ingreso (Pareto)
* Reino Unido concentra la mayoría de ventas
* El ingreso está impulsado por **volumen (Quantity)** más que por precio
* Productos de bajo precio y alta rotación dominan el negocio

---

## 🧠 Modelo

Se entrenó un modelo de **Random Forest Regressor** para predecir:

👉 `TotalPrice`

### Variables utilizadas:

* Quantity
* UnitPrice
* Year
* Month

El modelo fue registrado usando **MLflow**.

---

## 🚀 API de inferencia

Se implementó una API con FastAPI para exponer el modelo.

### Endpoint

```http
POST /predict
```

### Ejemplo de request

```json
{
  "Quantity": 10,
  "UnitPrice": 2.5,
  "Month": 12,
  "Year": 2011
}
```

### Ejemplo de respuesta

```json
{
  "prediction": 24.93,
  "currency": "GBP",
  "status": "success"
}
```

### Características

* Validación de datos con Pydantic
* Control de errores
* Consistencia de features usando `feature_names_in_`

---

## 🔄 Pipeline (Prefect)

Se implementó un pipeline que automatiza:

* Carga de datos
* Entrenamiento del modelo
* Registro en MLflow

---

## 🛠️ Tecnologías utilizadas

* Python
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* MLflow
* FastAPI
* Prefect

---

## ▶️ Ejecución

### 1. Crear entorno

```bash
python -m venv .venv
```

### 2. Activar entorno

```bash
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar API

```bash
uvicorn api:app --reload
```

### 5. Documentación

```bash
http://127.0.0.1:8000/docs
```

---

## 📌 Conclusión

Este proyecto demuestra la implementación de un flujo completo de MLOps, asegurando la transición desde el análisis exploratorio hasta el despliegue de un modelo en producción, garantizando consistencia entre entrenamiento y predicción.

---

## 👤 Autor

Clorena
