# TFG - Machine Learning

**Datos:** 01-01-2018 a 01-01-2024

**Descarga de datos:** https://anonfile.link/vvQNuCfE92U/data_rar

*Modelado:* 
Stacking (RF + SVM + KNN) 

*Meta-Modelo:* 
LogisticRegression.

## Estructura del proyecto

```
├── data/                # Carpeta ignorada por Git. Contiene los datos de entrada.
│   ├── COORDILLERA/     # Datos geoespaciales relacionados con la cordillera.
│   ├── DEM/             # Datos del modelo digital de elevación (DEM).
│   ├── FIRMS/           # Datos de incendios (FIRMS).
│   ├── Giovanni NASA/   # Datos obtenidos de la NASA (Giovanni).
│   └── NDVI/            # Índice de vegetación de diferencia normalizada.
├── outputs/             
├── .gitignore           
├── main.py              # Script principal que ejecuta todo el flujo: carga, preprocesa, entrena y predice.
├── clear_outputs.py     # Script de utilidad para limpiar la carpeta de resultados.
├── requirements.txt     # Lista de librerías de Python necesarias.
└── README.md            # Este archivo.

```


### Tabla de Variables del Modelo

#### **Variable Objetivo (Target)**

| Nombre de la Variable | Descripción | Tipo | Fuente de Datos Original | Formato/Extensión del Archivo | Valores Posibles |
| :-------------------- | :---------- | :--- | :----------------------- | :---------------------------- | :--------------- |
| **`fire`** | Indica si en un punto y fecha determinados ocurrió un incendio o no. | **Objetivo (Binaria)** | **Positivos:** FIRMS<br>**Negativos:** Muestreo aleatorio controlado | `.csv` | `1` (Incendio)<br>`0` (No Incendio) |

---
#### **Variables Predictoras (Features)**
| **Categoría** | **Nombre de la Variable** | **Descripción** | **Tipo** | **Fuente de Datos** | **Formato / Extensión** | **Unidad / Valores Ejemplo** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Topográficas** | `elevacion` | Altura sobre el nivel del mar. | Numérica | Modelo Digital de Elevación (SRTM, ALOS) | `.tif` | Metros (m) |
| | `pendiente` | Inclinación del terreno; influye en la velocidad de propagación del fuego. | Numérica | Derivado de `elevacion` | `.tif` | Grados (°) |
| | `orientacion_cat` | Dirección cardinal hacia la que se inclina la ladera. | Categórica | Derivado de `elevacion` | `.tif` | N, NE, E, SE, S, SW, W, NW, Plano |
| **Vegetación** | `ndvi` | Índice de Vegetación de Diferencia Normalizada. Representa la densidad y salud del combustible vegetal. | Numérica | Imágenes satelitales (MODIS, Landsat) | `.tif` | Rango: -1 a +1 |
| **Meteorológicas** | `temperature` | Temperatura del aire; afecta la sequedad del combustible. | Numérica | Giovanni (NASA) | `.csv` | Grados Celsius (°C) |
| | `precipitation` | Precipitación acumulada; reduce el riesgo al humedecer la vegetación y el suelo. | Numérica | Giovanni (NASA) | `.csv` | mm/hora |
| | `wind_speed` | Velocidad del viento; influye en la dirección e intensidad de la propagación del fuego. | Numérica | Giovanni (NASA) | `.csv` | Metros por segundo (m/s) |
| | `humedad` | Humedad relativa del aire. Valores altos dificultan la ignición y propagación del fuego. | Numérica | Giovanni (NASA) | `.tif` | Porcentaje (%) |
| **Proximidad / Antropogénica** | `dist_vias` | Distancia a la vía de comunicación principal más cercana. Puede indicar accesibilidad o fuente de ignición humana. | Numérica | Derivado de Shapefile de Vías | `.shp` | Metros (m) |
| | `dist_ciudades` | Distancia al centro poblado o ciudad más cercana. Relacionado con la actividad humana y posibles igniciones. | Numérica | Derivado de Shapefile de Ciudades | `.shp` | Metros (m) |


#### **Variables Intermedias (No usadas directamente en el modelo final, pero cruciales para la creación del dataset)**

| Nombre de la Variable | Descripción | Tipo | Fuente de Datos Original | Formato/Extensión del Archivo | Observaciones |
| :-------------------- | :---------- | :--- | :----------------------- | :---------------------------- | :-------------- |
| `geometry` | La coordenada (Punto) de cada observación. | Geoespacial | FIRMS y Muestreo aleatorio | (`.shp`, `.csv`) | Se usa para extraer valores de los rasters. |
| `date` | La fecha de cada observación. | Fecha/Hora | FIRMS y Muestreo aleatorio | `.csv` | Se usa para enlazar con NDVI y datos meteorológicos. |
| `orientacion` | La orientación numérica en grados (0-360) antes de ser categorizada. | Numérica | DEM (calculado) | `.tif` | Variable intermedia para crear `orientacion_cat`. |
| `(Polígono de área)` | Polígono que define los límites del área de estudio (Dpto. de Cordillera). | Geoespacial | Shapefile del departamento | `.shp` | Se usa para filtrar todos los datos espaciales. |

----



### Uso

1.  **Descargar los datos**

2.  **Preparar la carpeta de datos:**
    Descomprime el archivo `data.rar` y asegurar de que el contenido quede dentro de una carpeta llamada `data` en la raíz del proyecto. 

3.  **Ejecutar el modelo:**
    Para ejecutar el pipeline completo (desde la carga de datos hasta la generación del mapa de riesgo), ejecutar:
    ```bash
    python main.py
    ```
    Los resultados, incluyendo el reporte de clasificación y el mapa de riesgo, se guardarán en una nueva carpeta con fecha y hora dentro de `outputs/`.


### TFG (posible título): 
Predicción de riesgo de incendios forestales en el
departamento de Cordillera: Un enfoque basado
en stacking de modelos de Machine Learning con
datos geoespaciales