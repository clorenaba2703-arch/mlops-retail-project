# 🧠 MLOps Retail Project - EDA

## 📊 Descripción

Este proyecto presenta un Análisis Exploratorio de Datos (EDA) sobre el dataset **Online Retail**, con el objetivo de comprender patrones de ventas, comportamiento de clientes y características de los productos.

---

## 🎯 Objetivo

Explorar y analizar los datos para identificar:

- Tendencias de ventas
- Productos más vendidos
- Clientes más valiosos
- Distribución de precios
- Comportamiento geográfico

---

## 📁 Dataset

El dataset utilizado corresponde a transacciones de una tienda online, incluyendo información sobre:

- Facturas (`InvoiceNo`)
- Productos (`StockCode`, `Description`)
- Cantidad (`Quantity`)
- Precio (`UnitPrice`)
- Clientes (`CustomerID`)
- País (`Country`)
- Fecha (`InvoiceDate`)

---

## 🧹 Limpieza de datos

Se realizaron las siguientes transformaciones:

- Eliminación de valores nulos en `CustomerID`
- Eliminación de cancelaciones (facturas con "C")
- Eliminación de valores negativos en `Quantity` y `UnitPrice`

El dataset se redujo de **541,909 a 397,884 registros**.

---

## ⚙️ Feature Engineering

Se crearon nuevas variables:

- `TotalPrice`: valor total por transacción
- `Year`, `Month`, `Day`: variables temporales

---

## 📈 Análisis realizado

Se desarrollaron visualizaciones para:

- Ventas por mes
- Productos más vendidos
- Ventas por país
- Clientes más valiosos
- Distribución de precios (con y sin outliers)

---

## 💡 Principales insights

- Existe un comportamiento estacional en las ventas, con picos en los últimos meses del año.
- Un pequeño grupo de productos concentra la mayor parte de las ventas.
- El Reino Unido domina ampliamente las ventas.
- Un grupo reducido de clientes genera la mayor parte de los ingresos (principio de Pareto).
- La mayoría de los productos tienen precios bajos, con presencia de outliers en precios altos.

---

## 🛠️ Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## ▶️ Ejecución del proyecto

```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
jupyter notebook

AUTORES

---

# 🟢 PASO 2: Guardar archivo

👉 `Ctrl + S`

---

# 🟢 PASO 3: Subir a GitHub (MUY IMPORTANTE)

En la terminal:

```bash
git init
git add .
git commit -m "feat: complete EDA with data cleaning, visualization and insights"