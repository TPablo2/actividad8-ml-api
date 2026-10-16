from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

CARPETA = Path(__file__).resolve().parent
ruta_modelo = CARPETA / "modelo_iris.pkl"

datos_modelo = joblib.load(ruta_modelo)
modelo = datos_modelo["modelo"]
nombres_especies = datos_modelo["nombres_especies"]

@app.route("/", methods=["GET"])
def inicio():
    return jsonify({
        "mensaje": "API de clasificación de flores Iris",
        "estado": "Funcionando",
        "endpoint_prediccion": "/predict",
        "metodo": "POST"
    })

@app.route("/predict", methods=["POST"])
def predecir():
    datos = request.get_json(silent=True)

    if not isinstance(datos, dict):
        return jsonify({
            "error": "Debes enviar un objeto JSON."
        }), 400

    caracteristicas = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    if not all(campo in datos for campo in caracteristicas):
        return jsonify({
            "error": "Debes enviar las cuatro características de la flor.",
            "campos_requeridos": caracteristicas
        }), 400

    try:
        valores = [float(datos[campo]) for campo in caracteristicas]
    except (TypeError, ValueError):
        return jsonify({
            "error": "Todas las características deben ser números."
        }), 400

    if not np.all(np.isfinite(valores)):
        return jsonify({
            "error": "Los valores deben ser números finitos."
        }), 400

    if not all(0 < valor <= 10 for valor in valores):
        return jsonify({
            "error": "Las medidas deben ser mayores que 0 y menores o iguales a 10 cm."
        }), 400

    entrada = np.array(valores).reshape(1, 4)

    resultado = int(modelo.predict(entrada)[0])
    especie = nombres_especies[resultado]

    return jsonify({
        "prediccion": especie,
        "clase": resultado,
        "caracteristicas_recibidas": {
            campo: valor
            for campo, valor in zip(caracteristicas, valores)
        }
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
