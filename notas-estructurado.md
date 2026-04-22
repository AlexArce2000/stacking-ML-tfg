https://notebooklm.google.com/notebook/fd4efea0-bcbf-45f3-a34b-5c2e9858d18a
![alt text](<NotebookLM Mind Map.png>)


### Bloque 1: Adquisición y Extracción de Datos Multifuente
Este bloque ocurre principalmente en **Google Earth Engine (GEE)** y se encarga de obtener la materia prima.
*   **Fuentes:** FIRMS (Focos de calor), NASADEM (Topografía), ERA5-Land (Clima), MODIS (NDVI), ESA WorldCover (Suelo).
*   **Procesos Clave:**
    *   Muestreo puntual de píxeles (`reduceRegion`).
    *   Generación de muestras de control (No-incendios).
    *   Composición de mediana mensual (NDVI) y agregación diaria (Clima).
    *   Cálculo de variables rezagadas (Precipitación de 7 días).

### Bloque 2: Consolidación y Armonización Espacial
Este bloque ocurre en **Pandas** y unifica las diferentes fuentes en un solo dataset maestro.
*   **Procesos Clave:**
    *   **Estabilización espacial:** Redondeo de coordenadas a 4 decimales.
    *   **Fusión Integrada (Inner Join):** Unión de los 5 datasets por coordenadas y fecha.
    *   **Auditoría de Integridad:** Eliminación de duplicados y valores nulos (`dropna`).

### Bloque 3: Ingeniería de Variables Físico-Meteorológicas
Aquí es donde aplicas el conocimiento experto para que los datos tengan sentido científico.
*   **Procesos Clave:**
    *   **Magnitud del Viento:** Transformación vectorial $\sqrt{u^2 + v^2}$.
    *   **Humedad por Magnus:** Estimación de la humedad relativa mínima diaria.
    *   **Codificación Cíclica:** Transformación trigonométrica (Seno/Coseno) de los meses para representar la estacionalidad de Cordillera.

### Bloque 4: Pipeline de Machine Learning y Validación Temporal
Este bloque prepara los datos para los algoritmos y entrena los modelos.
*   **Procesos Clave:**
    *   **Validación Temporal:** División del dataset en Entrenamiento (2018-2022) y Prueba (2023).
    *   **Transformación Automática:** Uso de `ColumnTransformer` para escalado numérico (StandardScaler) y codificación categórica (One-Hot Encoding).
    *   **Entrenamiento Multi-modelo:** Ejecución de 7 algoritmos individuales (RF, XGB, LGBM, SVM, KNN, CAT, MLP).

### Bloque 5: Marco de Decisión Multicriterio (AHP-TOPSIS)
Este es el "corazón" de tu tesis, donde seleccionas el mejor modelo de forma científica.
*   **Procesos Clave:**
    *   **AHP (Ponderación):** Definición de la Matriz de Saaty priorizando el *Recall* y validación mediante el Ratio de Consistencia (CR).
    *   **Generación de Matriz de Decisión:** Recopilación de métricas (Accuracy, Recall, Tiempo, Memoria, etc.).
    *   **TOPSIS (Ranking):** Normalización vectorial y cálculo de distancias a la solución ideal para obtener el ranking final de modelos.

### Bloque 6: Visualización e Interpretación de Resultados
Fase final donde comunicas los hallazgos mediante gráficos técnicos.
*   **Procesos Clave:**
    *   **Gráficos de Desempeño:** Comparativa de barras de métricas clave.
    *   **Análisis de Discriminación:** Curvas ROC resaltando al ganador TOPSIS.
    *   **Análisis de Sensibilidad:** Gráfico de Importancia Relativa de Variables para explicar por qué ocurren los incendios en Cordillera.
