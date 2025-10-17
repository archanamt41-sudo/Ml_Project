# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np

# class PredictRequest(BaseModel):
#     data: list  # list of feature vectors or a single feature vector

# app = FastAPI(title='Iris RF Demo')

# # For demo, load a model saved earlier if present; otherwise create a dummy model
# try:
#     model = joblib.load('artifacts/rf_iris.joblib')
# except Exception as e:
#     # fallback: a simple scikit-learn model trained quickly
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split
#     iris = load_iris()
#     X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=10, random_state=42)
#     model.fit(X_train, y_train)

# @app.post('/predict')
# def predict(req: PredictRequest):
#     data = np.array(req.data)
#     # ensure 2D
#     if data.ndim == 1:
#         data = data.reshape(1, -1)
#     preds = model.predict(data).tolist()
#     return {'predictions': preds}
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Setup FastAPI
app = FastAPI(title='Iris RF Demo')

# Setup templates folder
templates = Jinja2Templates(directory="templates")

# Load model
try:
    model = joblib.load('artifacts/rf_iris.joblib')
except Exception as e:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

# Pydantic request
class PredictRequest(BaseModel):
    data: list

# API endpoint for programmatic prediction
@app.post('/predict')
def predict(req: PredictRequest):
    data = np.array(req.data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    preds = model.predict(data).tolist()
    return {'predictions': preds}

# HTML form page
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# Form submission endpoint
@app.post('/', response_class=HTMLResponse)
def predict_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    data = np.array(features).reshape(1, -1)
    pred = model.predict(data)[0]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": int(pred),
        "features": features
    })
