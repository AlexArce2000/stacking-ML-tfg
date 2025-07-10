# TFG - Machine Learning

**Datos:** 01-01-2018 a 01-01-2024
**Descarga de datos:** https://anonfile.link/vvQNuCfE92U/data_rar

*Modelado:* 
Stacking (RF + SVM + KNN) 

*Meta-Modelo:* 
LogisticRegression.
### Tabla de Variables del Modelo

#### **Variable Objetivo (Target)**

| Nombre de la Variable | Descripción | Tipo | Fuente de Datos Original | Formato/Extensión del Archivo | Valores Posibles |
| :-------------------- | :---------- | :--- | :----------------------- | :---------------------------- | :--------------- |
| **`fire`** | Indica si en un punto y fecha determinados ocurrió un incendio o no. | **Objetivo (Binaria)** | **Positivos:** FIRMS<br>**Negativos:** Muestreo aleatorio controlado | `.csv` | `1` (Incendio)<br>`0` (No Incendio) |

---
#### **Variables Predictoras (Features)**
|    **Categoría**   | **Nombre de la Variable** | **Descripción**                                                            | **Tipo**   | **Fuente de Datos**                      | **Formato / Extensión** | **Unidad / Valores Ejemplo**      |
| :----------------: | :------------------------ | -------------------------------------------------------------------------- | ---------- | ---------------------------------------- | ----------------------- | --------------------------------- |
|  **Topográficas**  | `elevacion`               | Altura sobre el nivel del mar.                                             | Numérica   | Modelo Digital de Elevación (SRTM, ALOS) | `.tif`                  | Metros (m)                        |
|                    | `pendiente`               | Inclinación del terreno; influye en la velocidad de propagación del fuego. | Numérica   | Derivado de `elevacion`                  | `.tif`                  | Grados (°)                        |
|                    | `orientacion_cat`         | Dirección cardinal de la ladera.                                           | Categórica | Derivado de `elevacion`                  | `.tif`                  | N, NE, E, SE, S, SW, W, NW, Plano |
|   **Vegetación**   | `ndvi`                    | Índice de vegetación que representa la densidad y salud del combustible.   | Numérica   | Imágenes satelitales (MODIS, Landsat)    | `.tif`                  | Rango: -1 a +1                    |
| **Meteorológicas** | `temperature`             | Temperatura del aire; afecta la sequedad del combustible.                  | Numérica   | Giovanni (NASA)                          | `.csv`                  | °C                                |
|                    | `precipitation`           | Precipitación; reduce el riesgo al humedecer la vegetación.                | Numérica   | Giovanni (NASA)                          | `.csv`                  | mm/hora                           |
|                    | `wind_speed`              | Velocidad del viento; influye en la propagación del fuego.                 | Numérica   | Giovanni (NASA)                          | `.csv`                  | m/s                               |

#### **Variables Intermedias (No usadas directamente en el modelo final, pero cruciales para la creación del dataset)**

| Nombre de la Variable | Descripción | Tipo | Fuente de Datos Original | Formato/Extensión del Archivo | Observaciones |
| :-------------------- | :---------- | :--- | :----------------------- | :---------------------------- | :-------------- |
| `geometry` | La coordenada (Punto) de cada observación. | Geoespacial | FIRMS y Muestreo aleatorio | (`.shp`, `.csv`) | Se usa para extraer valores de los rasters. |
| `date` | La fecha de cada observación. | Fecha/Hora | FIRMS y Muestreo aleatorio | `.csv` | Se usa para enlazar con NDVI y datos meteorológicos. |
| `orientacion` | La orientación numérica en grados (0-360) antes de ser categorizada. | Numérica | DEM (calculado) | `.tif` | Variable intermedia para crear `orientacion_cat`. |
| `(Polígono de área)` | Polígono que define los límites del área de estudio (Dpto. de Cordillera). | Geoespacial | Shapefile del departamento | `.shp` | Se usa para filtrar todos los datos espaciales. |

----



### Uso:

* Crear un entorno virtual
* Instalar los requerimientos:
```
pip install -r requirements.txt
``` 
* ejecutar:
```
python main.py
```


### TFG (posible título): 
Predicción de riesgo de incendios forestales en el
departamento de Cordillera: Un enfoque basado
en stacking de modelos de Machine Learning con
datos geoespaciales