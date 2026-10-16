from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

CARPETA = Path(__file__).resolve().parent

iris = load_iris()
X = iris.data
y = iris.target

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)

predicciones = modelo.predict(X_prueba)
exactitud = accuracy_score(y_prueba, predicciones)

print("MODELO DE CLASIFICACIÓN DE FLORES")
print("---------------------------------")
print(f"Cantidad total de muestras: {len(X)}")
print(f"Muestras para entrenamiento: {len(X_entrenamiento)}")
print(f"Muestras para prueba: {len(X_prueba)}")
print(f"Exactitud del modelo: {exactitud:.2%}")
print("\nReporte de clasificación:")
print(
    classification_report(
        y_prueba,
        predicciones,
        target_names=iris.target_names,
        zero_division=0
    )
)

modelo_final = DecisionTreeClassifier(random_state=42)
modelo_final.fit(X, y)

ruta_modelo = CARPETA / "modelo_iris.pkl"

joblib.dump(
    {
        "modelo": modelo_final,
        "nombres_especies": iris.target_names.tolist()
    },
    ruta_modelo
)

print(f"\nModelo guardado en: {ruta_modelo}")
