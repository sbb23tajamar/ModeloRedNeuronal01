from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
import joblib

app = Flask(__name__)

try:
    model = load_model('adam_model.h5', compile=False)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['mae', 'mse'])
    print("Modelo cargado y recompilado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise e

try:
    scaler = joblib.load('scaler.pkl')
    print("Escalador cargado correctamente.")
except FileNotFoundError:
    print("Error: El archivo 'scaler.pkl' no se encuentra en el directorio.")
    raise
except Exception as e:
    print(f"Error al cargar el escalador: {e}")
    raise e

scaler_features = [
    "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "2ndFlrSF", "GrLivArea", "BsmtFullBath",
    "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "WoodDeckSF",
    "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "MiscVal", "SalePrice"
]

model_features = [
    "OverallQual", "YearBuilt", "GrLivArea", "GarageCars", "GarageYrBlt",
    "YearRemodAdd", "FullBath", "TotalBsmtSF", "TotRmsAbvGrd", "Fireplaces",
    "LotArea", "WoodDeckSF"
]

@app.route('/predict-json', methods=['POST'])
def predict_json():
    try:
        json_data = request.get_json()
        if json_data is None:
            return jsonify({'error': 'No se encontró ningún JSON en la solicitud.'}), 400

        input_data = pd.DataFrame([json_data])

        # Rellenar columnas faltantes y ordenar según el escalador
        for feature in scaler_features:
            if feature not in input_data.columns:
                input_data[feature] = 0
        input_data = input_data[scaler_features]

        # Normalizar los datos
        input_data_scaled = scaler.transform(input_data)

        # Filtrar las características relevantes para el modelo
        input_data_filtered = input_data_scaled[:, [scaler_features.index(feature) for feature in model_features]]

        # Realizar las predicciones
        predictions = model.predict(input_data_filtered)

        # Crear la respuesta JSON con las columnas relevantes y el precio predicho
        results = input_data[model_features].copy()
        results['PredictedPrice'] = predictions.flatten()
        return results.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)