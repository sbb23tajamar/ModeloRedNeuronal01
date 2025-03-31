import requests
import pandas as pd

# URL de la API
url = "http://127.0.0.1:5000/predict-json"

# JSON original con todos los datos
original_data = {
    "Id":1468,
        "MSSubClass":60,
        "MSZoning":"RL",
        "LotFrontage":63.0,
        "LotArea":8402,
        "Street":"Pave",
        "Alley":"null",
        "LotShape":"IR1",
        "LandContour":"Lvl",
        "Utilities":"AllPub",
        "LotConfig":"Inside",
        "LandSlope":"Gtl",
        "Neighborhood":"Gilbert",
        "Condition1":"Norm",
        "Condition2":"Norm",
        "BldgType":"1Fam",
        "HouseStyle":"2Story",
        "OverallQual":6,
        "OverallCond":5,
        "YearBuilt":1998,
        "YearRemodAdd":1998,
        "RoofStyle":"Gable",
        "RoofMatl":"CompShg",
        "Exterior1st":"VinylSd",
        "Exterior2nd":"VinylSd",
        "MasVnrType":"None",
        "MasVnrArea":0.0,
        "ExterQual":"TA",
        "ExterCond":"TA",
        "Foundation":"PConc",
        "BsmtQual":"Gd",
        "BsmtCond":"TA",
        "BsmtExposure":"No",
        "BsmtFinType1":"Unf",
        "BsmtFinSF1":0.0,
        "BsmtFinType2":"Unf",
        "BsmtFinSF2":0.0,
        "BsmtUnfSF":789.0,
        "TotalBsmtSF":789.0,
        "Heating":"GasA",
        "HeatingQC":"Gd",
        "CentralAir":"Y",
        "Electrical":"SBrkr",
        "1stFlrSF":789,
        "2ndFlrSF":676,
        "LowQualFinSF":0,
        "GrLivArea":1465,
        "BsmtFullBath":0.0,
        "BsmtHalfBath":0.0,
        "FullBath":2,
        "HalfBath":1,
        "BedroomAbvGr":3,
        "KitchenAbvGr":1,
        "KitchenQual":"TA",
        "TotRmsAbvGrd":7,
        "Functional":"Typ",
        "Fireplaces":1,
        "FireplaceQu":"Gd",
        "GarageType":"Attchd",
        "GarageYrBlt":1998.0,
        "GarageFinish":"Fin",
        "GarageCars":2.0,
        "GarageArea":393.0,
        "GarageQual":"TA",
        "GarageCond":"TA",
        "PavedDrive":"Y",
        "WoodDeckSF":0,
        "OpenPorchSF":75,
        "EnclosedPorch":0,
        "3SsnPorch":0,
        "ScreenPorch":0,
        "PoolArea":0,
        "PoolQC":"null",
        "Fence":"null",
        "MiscFeature":"null",
        "MiscVal":0,
        "MoSold":5,
        "YrSold":2010,
        "SaleType":"WD",
        "SaleCondition":"Normal"
}

# Características esperadas por el escalador
scaler_features = [
    "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "2ndFlrSF", "GrLivArea", "BsmtFullBath",
    "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "WoodDeckSF",
    "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "MiscVal", "SalePrice"
]

# Características relevantes para el modelo
model_features = [
    "OverallQual", "YearBuilt", "GrLivArea", "GarageCars", "GarageYrBlt",
    "YearRemodAdd", "FullBath", "TotalBsmtSF", "TotRmsAbvGrd", "Fireplaces",
    "LotArea", "WoodDeckSF"
]

# Convertir a DataFrame para manejar las columnas
df = pd.DataFrame([original_data])

# Rellenar las columnas faltantes para el escalador con valores predeterminados
for feature in scaler_features:
    if feature not in df.columns:
        df[feature] = 0  # Rellenar valores faltantes con 0

# Filtrar solo las columnas necesarias para el escalador, en el orden correcto
filtered_data = df[scaler_features].to_dict(orient="records")[0]

try:
    # Enviar la solicitud POST con los datos JSON filtrados
    response = requests.post(url, json=filtered_data)

    # Verificar el código de estado de la respuesta
    if response.status_code == 200:
        # Mostrar la respuesta de la API (predicciones)
        print("Respuesta de la API:")
        print(response.json())
    else:
        print(f"Error en la solicitud. Código de estado: {response.status_code}")
        print("Detalles del error:", response.text)
except Exception as e:
    print(f"Ocurrió un error: {e}")