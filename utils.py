import pandas as pd
import numpy as np
#import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData
import laspy
from sklearn.preprocessing import StandardScaler


def read_las_file(file_path):
    """
    Lee un archivo LAS (formato de archivo comúnmente utilizado para datos LiDAR) y devuelve los puntos.

    Args:
        file_path (str): La ruta del archivo LAS que se desea leer.

    Returns:
        numpy.ndarray or None: Un arreglo NumPy que contiene los puntos leídos del archivo LAS,
        o None si ocurre un error durante la lectura del archivo.
    """
    import pylas

    try:
        # Intenta leer el archivo LAS utilizando la biblioteca pylas.
        las = pylas.read(file_path)
        return las.points
    except Exception as e:
        # En caso de error, imprime un mensaje de error y devuelve None.
        print(f"Error al leer {file_path}: {e}")
        return None

# descomentar en caso de ser usada
"""def lod_mesh_export_from_csv(csv_path, lods, name, extension, output_path):
    
    Lee las coordenadas de un archivo CSV y exporta diferentes niveles de detalle (LoDs) de una malla a archivos utilizando
    la técnica de simplificación de decimación cuádrica.

    Args:
        csv_path (str): La ruta del archivo CSV que contiene las coordenadas de la malla.
        lods (list): Una lista de enteros que representa los niveles de detalle deseados.
        extension (str): La extensión del archivo para guardar la malla (por ejemplo, '.ply').
        output_path (str): La ruta donde se guardarán los archivos exportados.

    Returns:
        dict: Un diccionario que mapea los niveles de detalle (LoDs) a sus respectivas mallas simplificadas.
    
    # Leer coordenadas desde el archivo CSV
    df = pd.read_csv(csv_path)
    vertices = df[['X', 'Y', 'Z']].values

    # Crear una malla TriangleMesh a partir de las coordenadas
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    mesh_lods = {}  # Un diccionario para almacenar las mallas simplificadas por nivel de detalle.

    for i in lods:
        # Simplificar la malla utilizando decimación cuádrica para el nivel de detalle actual.
        mesh_lod = mesh.simplify_quadric_decimation(i)

        # Escribir la malla simplificada en un archivo utilizando el formato especificado.
        o3d.io.write_triangle_mesh(output_path + f"lod_{name}_" + str(i) + extension, mesh_lod)

        # Agregar la malla simplificada al diccionario de LoDs.
        mesh_lods[i] = mesh_lod

    # Imprimir un mensaje de éxito para la generación de niveles de detalle.
    print("Generación de niveles de detalle exitosa.")

    # Devolver el diccionario de mallas simplificadas por nivel de detalle.
    return mesh_lods
"""


def plot_point_cloud(dataframe):
    # Proyecta los puntos en un gráfico 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(dataframe['x'], dataframe['y'], c=dataframe[['red', 'green', 'blue']].values / 255.0, s=1)

    # Personaliza el gráfico según tus necesidades
    plt.title('Nube de Puntos 2D')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True)

    # Agrega una barra de colores
    cbar = plt.colorbar()
    cbar.set_label('RGB Color')

    # Muestra el gráfico
    plt.show()

    # Limpia la figura y libera la memoria
    plt.close()
    del dataframe


def plot_point_cloud_xyz(dataframe):
    # Proyecta los puntos en un gráfico 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(dataframe['x'], dataframe['y'], cmap='viridis', s=1)

    # Personaliza el gráfico según tus necesidades
    plt.title('Nube de Puntos 2D')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True)

    # Agrega una barra de colores
    cbar = plt.colorbar()
    cbar.set_label('Coordenada Z')

    # Muestra el gráfico
    plt.show()

    # Limpia la figura y libera la memoria
    plt.close()
    del dataframe


def ply_to_dataframe(file_path):
    try:
        # Abrir el archivo PLY
        plydata = PlyData.read(file_path)

        # Obtener las coordenadas x, y, z de los vértices
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        red = plydata['vertex']['red']
        green = plydata['vertex']['green']
        blue = plydata['vertex']['blue']
        classe = plydata['vertex']['class']
        #instance = plydata['vertex']['instance']
        # Crear un DataFrame de Pandas
        df = pd.DataFrame({'x': x, 'y': y, 'z': z,'red':red,'green':green,'blue':blue,'class':classe})

        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def convert_las_to_csv(las_file_path, output_csv_path, attribute_name, batch_size=100000):
    """
    Carga un archivo LAS, extrae coordenadas XYZ y un atributo específico,
    crea un DataFrame de pandas y guarda los datos en un archivo CSV.

    Parámetros:
    las_file_path (str): Ruta del archivo LAS a cargar.
    output_csv_path (str): Ruta para guardar el archivo CSV resultante.
    attribute_name (str): Nombre del atributo a extraer del archivo LAS.
    batch_size (int): Tamaño del lote para procesar en cada iteración.

    Ejemplo de uso:
    convert_las_to_csv('/content/datoslas/n1_sin camino.las', 'archivo.csv', 'intensity')
    """
    las = laspy.read(las_file_path)

    xyz_cols = ['X', 'Y', 'Z']
    attribute_cols = [attribute_name]

    num_points = len(las.points)

    for i in range(0, num_points, batch_size):
        end_idx = min(i + batch_size, num_points)
        batch = las.points[i:end_idx]

        xyz_data = {dim: batch[dim] for dim in xyz_cols}
        attribute_data = {attribute_name: batch[attribute_name]}

        if i == 0:
            df = pd.DataFrame({**xyz_data, **attribute_data})
        else:
            df_batch = pd.DataFrame({**xyz_data, **attribute_data})
            df = pd.concat([df, df_batch], ignore_index=True)

    df.to_csv(output_csv_path, index=False)


def test_model(data, model, target_value=6):
    """
    Evalúa un modelo de machine learning en un conjunto de datos y visualiza las predicciones.

    Parámetros:
        - data (DataFrame): El conjunto de datos de entrada que contiene al menos tres columnas ('x', 'y', 'z').
        - model: El modelo de machine learning que se va a evaluar.
        - target_value (int): El valor objetivo para filtrar las predicciones. Predicciones igual a este valor se incluirán en la visualización.

    Retorna:
        None
    """
    # Obtener las columnas x, y, y z del conjunto de datos
    d = data.iloc[:, :3]

    # Estandarizar las características
    scaler = StandardScaler()
    X_ = scaler.fit_transform(d)

    # Realizar predicciones con el modelo
    pred = model.predict(X_)

    # Crear un DataFrame con las características originales, predicciones y filtrar por valor objetivo
    df = pd.DataFrame({'x': X_[:, 0], 'y': X_[:, 1], 'z': X_[:, 2], 'predicciones': pred})
    df = df[df['predicciones'] == target_value]

    # Visualizar el punto en la nube XYZ
    plot_point_cloud_xyz(df)