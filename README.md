# API para modelo de datos lidar (ply, las)

En el codigo de este proyecto se creo una API usando Flask para obtener las predicciones de un modelo XGboost al pasarle las coordenadas x,y,z de datos lidar en formato JSON. 

## SETUP

**Instala virtualenv en CMD usando pip:**
   ```bash
   pip install virtualenv
   ```

### Crear y Activar Entorno Virtual

Ejecuta los siguientes comandos para crear y activar un entorno virtual (puedes elegir otro nombre si lo deseas):

```bash
# Crear entorno virtual (reemplaza 'myenv' con el nombre que desees)
virtualenv myenv
```
```bash
# Activar entorno virtual (en Windows)
.\myenv\Scripts\activate
```

Después de ejecutar el segundo comando, verás el nombre de tu entorno virtual en el indicador de la línea de comandos, indicando que el entorno virtual está activo. Por ejemplo:

```bash
(myenv) C:\ruta\a\tu\proyecto>
```

Cuando hayas terminado de trabajar en tu entorno virtual, puedes desactivarlo con el siguiente comando:

```bash
deactivate
```
Ahora estás listo para trabajar en tu entorno virtual de Python en Windows 10. Recuerda que cada vez que necesites trabajar en tu proyecto, primero debes activar el entorno virtual utilizando el comando de: 
**\myenv\Scripts\activate**

Proceder con la ejecucion del siguiente comando para instalar las dependencias necesarias para que el proyecto funcione

```bash
pip install -r requirements.txt
```
**importante** que el modelo llamado **cat-model-hans.json** este en el mismo directorio que **main.py**. otra aclaracion es que utils.py es un archivo con funciones utiles que pueden servir ya que se usaron para entrenar el modelo xgboost, de momento en la API no se usaron explicitamente

### muchas gracias por su confianza en mi, cuenten conmigo ante cualquier duda o inconveniente con la app. Atte. David