import requests

url = "http://127.0.0.1:5000/predict"

datos_flor = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

try:
    respuesta = requests.post(
        url,
        json=datos_flor,
        timeout=10
    )

    print("Código de respuesta:", respuesta.status_code)
    print("Resultado de la predicción:")
    print(respuesta.text)

except requests.exceptions.RequestException as error:
    print("Error al conectar con la API:")
    print(error)
