from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
datos_modelo = joblib.load("modelo_iris.pkl")

# Obtener el modelo y los nombres de las especies
modelo = datos_modelo["modelo"]
nombres_especies = datos_modelo["nombres_especies"]


@app.route("/")
def inicio():
    # Mostrar la página web
    return render_template("index.html")


@app.route("/health")
def estado():
    # Consultar el estado de la API
    return jsonify({
        "estado": "Funcionando",
        "mensaje": "API de clasificación de flores Iris"
    })


@app.route("/predict", methods=["POST"])
def predecir():
    try:
        datos = request.get_json()

        if not datos:
            return jsonify({
                "error": "No se recibieron datos JSON."
            }), 400

        variables = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]

        if not all(variable in datos for variable in variables):
            return jsonify({
                "error": "Faltan una o más medidas de la flor."
            }), 400

        medidas = [float(datos[variable]) for variable in variables]

        if not all(np.isfinite(valor) and valor > 0 for valor in medidas):
            return jsonify({
                "error": "Todas las medidas deben ser números positivos."
            }), 400

        entrada = np.array(medidas).reshape(1, -1)

        prediccion = modelo.predict(entrada)[0]
        especie = str(nombres_especies[int(prediccion)])

        return jsonify({
            "especie": especie,
            "prediccion": especie,
            "medidas": {
                variable: valor
                for variable, valor in zip(variables, medidas)
            }
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Las medidas deben ser números válidos."
        }), 400

    except Exception:
        app.logger.exception("Error al realizar la predicción")
        return jsonify({
            "error": "Ocurrió un error al procesar la predicción."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
