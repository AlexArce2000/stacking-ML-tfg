# TFG - Machine Learning

**Datos:** 01-01-2018 a 31-12-2023

**Descarga de datos:** 
* https://anonfile.link/s/G6Bh7gcdmiL

* https://drive.google.com/drive/folders/1E2Wbdy16W65BfXifNLOEWrCAh_Y_lrQQ?usp=drive_link

**Dashboard básico**: https://alexarce2000.github.io/stacking-ML-tfg/

*Modelado:* 
Stacking (RF + SVM + KNN) 

*Meta-Modelo:* 
LogisticRegression.

**Vizualizr notebook en la web creando un .html:** 
```
jupyter nbconvert --to html .\main.ipynb
```

## Estructura del proyecto

```
.
├── data/                       # Carpeta con todos los datos de entrada brutos.
│   ├── Cobertura_del_Suelo/    # Datos geoespaciales de cobertura (bosque, pastizal, etc.).
│   ├── COORDILLERA/            # Shapefiles y datos vectoriales de la zona de estudio.
│   ├── DEM/                    # Datos del Modelo Digital de Elevación.
│   ├── FIRMS/                  # Datos de focos de calor de la NASA.
│   ├── Humedad/                # Datos de humedad relativa.
│   ├── NDVI/                   # Datos del Índice de Vegetación de Diferencia Normalizada.
│   ├── Precipitation/          # Datos de precipitación.
│   ├── Temperature/            # Datos de temperatura del aire.
│   └── Wind/                   # Datos de velocidad del viento.
│
├── doc/                        # Carpeta para la documentación.
│   └── documentacion_simple/   # Archivos LaTeX, mini documentación.
│
├── frontend/                   # Frontend básico para el dashboard de resultados.
│   ├── output/                 # Últimos resultados copiados desde las carpetas 'outputs'.
│   ├── js/                     # Archivos JavaScript para la interactividad.
│   ├── style/                  # Archivos CSS para los estilos.
│   └── variables.html          # (ejemplo de página del dashboard)
│
├── outputs/                    # Carpeta principal de resultados del pipeline de predicción (main.py).
├── outputs_analisis_exploratorio/ # Resultados del Análisis Exploratorio de Datos (generados por generar_graficos_eda.py).
├── outputs_evaluacion_modelo/  # Gráficos de evaluación avanzada del modelo (generados por generar_graficos_evaluacion.py).
├── outputs_focos_reales/       # (Carpeta existente para reportes de focos).
├── outputs_grandes_incendios/  # Resultados del script que identifica grandes incendios (generado por grandes_incendios.py).
│
├── .gitignore                  # Archivo que define qué ficheros y carpetas ignorar en Git.
├── clear_outputs.py            # Script de utilidad para limpiar las carpetas de resultados.
├── datos_depurados_para_eda.pkl # Archivo intermedio generado por main.py para ser usado por el script de EDA.
├── focos_cordillera.py         # Script de utilidad para reportar lo que realmente sucedió.
├── focos_cordillera_historico_completo.py # Script de utilidad para reportar el total de incendios 2018-2023.
├── generar_graficos_eda.py     # Script dedicado a generar los gráficos del Análisis Exploratorio de Datos.
├── generar_graficos_evaluacion.py # Script dedicado a generar los gráficos de evaluación avanzada del modelo.
├── grandes_incendios.py        # Script que implementa el algoritmo para identificar grandes incendios.
├── index.html                  # Página principal para el dashboard del frontend.
├── main.py                     # Script principal que ejecuta el flujo: carga, preprocesa, entrena, predice y evalúa.
├── main.ipynb                  # Notebook del proyecto con análisis y visualizaciones interactivas.
├── main.html                   # Versión HTML del notebook para compartir fácilmente.
├── notas.md                    # Notas y apuntes personales sobre el proyecto.
├── README.md                   # Descripción general del proyecto.
├── requirements.txt            # Lista de librerías de Python necesarias (ej. pandas, sklearn).
└── resultados_modelo_para_eval.pkl # Archivo intermedio con el modelo y datos de prueba para la evaluación avanzada.
```


### Tabla de Variables del Modelo

#### **Variable Objetivo (Target)**

| Nombre de la Variable | Descripción | Tipo | Fuente de Datos Original | Formato/Extensión del Archivo | Valores Posibles |
| :-------------------- | :---------- | :--- | :----------------------- | :---------------------------- | :--------------- |
| **`fire`** | Indica si en un punto y fecha determinados ocurrió un incendio o no. | **Objetivo (Binaria)** | **Positivos:** FIRMS<br>**Negativos:** Muestreo aleatorio controlado | `.csv` | `1` (Incendio)<br>`0` (No Incendio) |

---
---

### **Tabla de Variables Actualizada**

#### **Variables Predictoras (Features)**

| **Categoría** | **Nombre de la Variable** | **Descripción** | **Tipo** | **Fuente de Datos** | **Formato / Extensión** | **Unidad / Valores Ejemplo** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Topográficas** | `elevacion` | Altura sobre el nivel del mar. | Numérica | Modelo Digital de Elevación (DEM) | `.tif` | Metros (m) |
| | `pendiente` | Inclinación del terreno; influye en la velocidad de propagación del fuego. | Numérica | Derivado de `elevacion` | `.tif` | Grados (°) |
| | `orientacion_cat` | Dirección cardinal hacia la que se inclina la ladera (aspecto). | Categórica | Derivado de `elevacion` | `.tif` | N, NE, E, SE, S, SW, W, NW, Plano |
| **Uso del Suelo**| `cobertura` | Tipo de cobertura del suelo (bosque, pastizal, etc.). Factor clave del tipo y cantidad de combustible. | Categórica | Clasificación de Cobertura del Suelo | `.tif` | Códigos de clase (ej: 3.0, 11.0, 12.0) |
| **Vegetación** | `ndvi` | Índice de Vegetación de Diferencia Normalizada. Representa la densidad y salud del combustible vegetal. | Numérica | Imágenes satelitales (MODIS vía GEE) | `.tif` | Rango normalizado: -1 a +1 |
| **Meteorológicas**| `temperature` | Temperatura media diaria del aire. | Numérica | Datos Climáticos (ERA5 vía GEE) | `.csv` | Grados Celsius (°C) |
| | `precipitation` | Precipitación total diaria. | Numérica | Datos Climáticos (ERA5 vía GEE) | `.csv` | Milímetros (mm/día) |
| | `wind_speed` | Velocidad media diaria del viento. | Numérica | Datos Climáticos (ERA5 vía GEE) | `.csv` | Metros por segundo (m/s) |
| | `humidity` | Humedad relativa media diaria del aire. | Numérica | Datos Climáticos (ERA5 vía GEE) | `.csv` | Porcentaje (%) |
| **Proximidad** | `dist_vias` | Distancia a la vía de comunicación principal más cercana. | Numérica | Shapefile de Vías | `.shp` | Metros (m) |
| | `dist_ciudades`| Distancia al centro poblado o ciudad más cercana. | Numérica | Shapefile de Ciudades | `.shp` | Metros (m) |
| **Objetivo** | `fire` | Variable objetivo que indica la presencia (1) o ausencia (0) de un incendio. | Binaria | Puntos de calor (FIRMS) y Muestreo | `.csv` | 0 (No Incendio), 1 (Incendio) |

#### **Variables Intermedias (No usadas directamente en el modelo final, pero cruciales para la creación del dataset)**

| Nombre de la Variable | Descripción | Tipo | Fuente de Datos Original | Formato/Extensión | Observaciones |
| :-------------------- | :---------- | :--- | :----------------------- | :------------------ | :-------------- |
| `geometry` | La coordenada (Punto) de cada observación. | Geoespacial | FIRMS y Muestreo de fondo | (`.csv`, GeoDataFrame) | Se usa para extraer valores de los rasters y calcular distancias. |
| `date` | La fecha de cada observación. | Fecha/Hora | FIRMS y Muestreo de fondo | `.csv` | Se usa para enlazar con NDVI y datos meteorológicos diarios. |
| `orientacion` | La orientación numérica en grados (0-360) antes de ser categorizada. | Numérica | Derivado de `elevacion` | En memoria | Variable intermedia para crear `orientacion_cat`. |
| `(Polígono de área)` | Polígono que define los límites del área de estudio (Dpto. de Cordillera). | Geoespacial | Shapefile del departamento | `.shp` | Se usa para filtrar todos los datos espaciales. |
|`zonas_inflamables_poligono`| Polígono que define las áreas consideradas inflamables. | Geoespacial | Derivado de `cobertura` | En memoria | Se usa para generar los puntos de "No Incendio" de forma inteligente. |

---



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


Nota: Agregado el algoritmo verficador de la realidad `focos_cordillera.py`, que reporta lo que realmente sucedió en fechas especificas. 

### TFG (posible título): 
Predicción de riesgo de incendios forestales en el
departamento de Cordillera: Un enfoque basado
en stacking de modelos de Machine Learning con
datos geoespaciales