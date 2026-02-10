from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import shutil
import os

from ml.sst_model import train_and_predict_sst
from ml.fisheries_model import train_and_predict_catch
from ml.biodiversity_model import predict_dominant_species
from ml.biodiversity_risk_model import predict_species_risk


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ ANALYSIS ------------------

@app.post("/analyze/fisheries")
async def analyze_fisheries(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    required_cols = ["catch_kg", "species", "water_temp"]

    for col in required_cols:
        if col not in df.columns:
            return {"error": f"{col} column missing"}

    return {
        "total_catch_kg": float(df["catch_kg"].sum()),
        "average_water_temperature": float(df["water_temp"].mean()),
        "species_catch": df.groupby("species")["catch_kg"].sum().to_dict()
    }

@app.post("/analyze/ocean")
async def analyze_ocean(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "sst" not in df.columns:
        return {"error": "sst column missing"}

    return {
        "summary": {
            "sst": df["sst"].describe().to_dict()
        }
    }

@app.post("/analyze/biodiversity")
async def analyze_biodiversity(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {
        "species_count": int(df["species"].nunique()),
        "distribution": df["species"].value_counts().to_dict()
    }

# ------------------ PREDICTION ------------------

@app.post("/predict/sst")
async def predict_sst(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction = train_and_predict_sst(temp_path)
    finally:
        os.remove(temp_path)

    return {"predicted_sst": prediction}

@app.post("/predict/fisheries")
async def predict_fisheries(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction = train_and_predict_catch(temp_path)
    finally:
        os.remove(temp_path)

    return {"predicted_catch_kg": prediction}

@app.post("/predict/biodiversity")
async def predict_biodiversity(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        dominant_species = predict_dominant_species(temp_path)
    finally:
        os.remove(temp_path)

    return {"predicted_dominant_species": dominant_species}

@app.post("/predict/biodiversity/risk")
async def predict_biodiversity_risk(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_species_risk(temp_path)
        return {"risk_analysis": result}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

