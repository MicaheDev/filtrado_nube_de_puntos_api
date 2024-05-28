import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import laspy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


def preprocesar(data):
    scaler = StandardScaler()
    X_ = scaler.fit_transform(data)
    matrix_data = xgb.DMatrix(X_)
    return matrix_data


class Hello(Resource):
    def get(self):
        return "Welcome to Test App API!"

    def post(self):
        try:
            value = request.get_json()
            if value:
                return {"Post Values": value}, 201
            return {"error": "Invalid format."}
        except Exception as error:
            return {"error": str(error)}


class Predict(Resource):
    def get(self):
        return {"error": "Invalid Method."}

    def post(self):
        try:
            if "file" not in request.files:
                return (
                    jsonify({"error": "No se ha proporcionado ningún archivo .las"}),
                    400,
                )

            archivo = request.files["file"]
            if archivo.filename == "":
                return jsonify({"error": "Nombre de archivo no válido"}), 400

            if archivo and archivo.filename.endswith(".las"):
                archivo_path = guardar_archivo_temporal(archivo)
                in_file = laspy.read(archivo_path)
                header = in_file.header

                coordinates, red_values, green_values, blue_values, intensity_values = (
                    obtener_atributos(in_file)
                )
                df = pd.DataFrame(coordinates)

                if not all(
                    len(lst) == len(coordinates["x"]) for lst in coordinates.values()
                ):
                    raise ValueError("All coordinate lists must have the same length.")

                clases_predichas = predecir_clases(df)

                # print(clases_predichas)

                # Filtrar las coordenadas y atributos basados en las clases predichas (solo clase 11)
                indices_validos = [
                    i for i, clase in enumerate(clases_predichas) if clase in [12,15]
                ]
                if not indices_validos:
                    return (
                        jsonify(
                            {
                                "error": "No hay puntos que coincidan con la clase predicha (11)"
                            }
                        ),
                        400,
                    )

                coordenadas_filtradas = {
                    key: np.array(val)[indices_validos]
                    for key, val in coordinates.items()
                }
                red_filtradas = (
                    np.array(red_values)[indices_validos]
                    if red_values is not None
                    else None
                )
                green_filtradas = (
                    np.array(green_values)[indices_validos]
                    if green_values is not None
                    else None
                )
                blue_filtradas = (
                    np.array(blue_values)[indices_validos]
                    if blue_values is not None
                    else None
                )
                intensity_filtradas = (
                    np.array(intensity_values)[indices_validos]
                    if intensity_values is not None
                    else None
                )

                new_las = crear_nuevo_las(
                    header,
                    coordenadas_filtradas,
                    red_filtradas,
                    green_filtradas,
                    blue_filtradas,
                    intensity_filtradas,
                )

                new_las_path = guardar_nuevo_las(new_las)

                limpiar_carpeta_archivos_guardados()


                return enviar_archivo(new_las_path, "new.las")
        except Exception as error:
            return jsonify({"error": str(error)})


def guardar_archivo_temporal(archivo):
    directorio_actual = os.getcwd()
    directorio_guardado = os.path.join(directorio_actual, "archivos_guardados")
    if not os.path.exists(directorio_guardado):
        os.makedirs(directorio_guardado)
    archivo_path = os.path.join(directorio_guardado, archivo.filename)
    archivo.save(archivo_path)
    return archivo_path


def obtener_atributos(in_file):
    x_values = np.array(in_file.x).tolist()
    y_values = np.array(in_file.y).tolist()
    z_values = np.array(in_file.z).tolist()
    red_values = np.array(in_file.red).tolist() if hasattr(in_file, "red") else None
    green_values = (
        np.array(in_file.green).tolist() if hasattr(in_file, "green") else None
    )
    blue_values = np.array(in_file.blue).tolist() if hasattr(in_file, "blue") else None
    intensity_values = (
        np.array(in_file.intensity).tolist() if hasattr(in_file, "intensity") else None
    )
    coordinates = {"x": x_values, "y": y_values, "z": z_values}
    return coordinates, red_values, green_values, blue_values, intensity_values


def predecir_clases(df):
    booster = xgb.Booster()
    booster.load_model("cat-model-hans.json")
    DATA_PROCESADA = preprocesar(df)
    pred = booster.predict(DATA_PROCESADA)
    mapeo_clases = {0: 1, 1: 2, 2: 5, 3: 11, 4: 13, 5: 14, 6: 15, 7: 18, 8: 19}
    clases_predichas = [mapeo_clases[np.argmax(fila)] for fila in pred]
    return clases_predichas


def crear_nuevo_las(
    header, coordinates, red_values, green_values, blue_values, intensity_values
):
    new_las = laspy.create(file_version="1.2", point_format=3)

    new_las.X = coordinates["x"]
    new_las.Y = coordinates["y"]
    new_las.Z = coordinates["z"]

    if red_values is not None:
        new_las.red = red_values
    if green_values is not None:
        new_las.green = green_values
    if blue_values is not None:
        new_las.blue = blue_values
    if intensity_values is not None:
        new_las.intensity = intensity_values

    # print("x antes de retornar:", new_las.x)
    # print("y antes de retornar:", new_las.y)
    # print("z antes de retornar:", new_las.z)
    return new_las


def guardar_nuevo_las(new_las):
    temp_dir = tempfile.mkdtemp()
    new_las_path = os.path.join(temp_dir, "new_file.las")
    new_las.write(new_las_path)
    return new_las_path


def enviar_archivo(path, name):
    return send_file(
        path, 
        as_attachment=True, 
        download_name=name, 
        mimetype='application/octet-stream'
    )

def limpiar_carpeta_archivos_guardados():
    directorio_guardado = os.path.join(os.getcwd(), "archivos_guardados")
    if os.path.exists(directorio_guardado):
        shutil.rmtree(directorio_guardado)

api.add_resource(Hello, "/")
api.add_resource(Predict, "/predict")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
