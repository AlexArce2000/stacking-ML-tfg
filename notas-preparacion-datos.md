### 1. Estabilización de Coordenadas (Redondeo de Precisión)
*   **Técnica:** Redondeo de punto flotante a 4 decimales.
*   **Definición:** Ajuste de la precisión numérica de las coordenadas geográficas para estandarizar el punto de referencia.
*   **Dónde se usó:** En la función `estabilizar_coordenadas(df)` antes de realizar el `merge` de los datasets.
*   **Justificación:** Los datos satelitales (ERA5, NASADEM, MODIS) tienen ligeras variaciones de precisión. El redondeo a 4 decimales (aprox. 11 metros de precisión) asegura que los datos de diferentes fuentes "encajen" perfectamente en el mismo píxel sin perder la ubicación real.

### 2. Transformación de Magnitud del Viento (Fórmula de Pitágoras)
*   **Técnica:** Conversión de componentes vectoriales $U$ y $V$ a Magnitud Escalar.
*   **Definición:** Cálculo de la hipotenusa de un vector resultante a partir de sus componentes ortogonales mediante la fórmula $\sqrt{u^2 + v^2}$.
*   **Dónde se usó:** En el bloque de **Ingeniería de Variables** (Pág. 44 del PDF y función `preparar_dataset_tesis`).
*   **Justificación:** Los modelos de Machine Learning interpretan mejor la intensidad del viento como una sola magnitud física que como dos vectores separados. Cumple con el requisito técnico de tu marco teórico.

### 3. Cálculo de Humedad Relativa (Fórmula de Magnus)
*   **Técnica:** Estimación de Humedad Relativa mediante presión de vapor de saturación.
*   **Definición:** Relación entre la presión de vapor real (punto de rocío) y la presión de vapor saturada (temperatura del aire).
*   **Dónde se usó:** En el script de Python para procesar el archivo `Dataset_Cordillera_Meteorologia.csv`.
*   **Justificación:** La humedad relativa no es una variable directa en ERA5-Land. Su cálculo es vital porque, como demostró tu gráfico de importancia, es el predictor #1 del riesgo de incendio en Cordillera.

### 4. Codificación Cíclica del Tiempo (Seno y Coseno)
*   **Técnica:** Transformación Trigonométrica de variables temporales.
*   **Definición:** Mapeo de valores cíclicos (como los 12 meses del año) a coordenadas en un círculo unitario.
*   **Dónde se usó:** En la función `preparar_dataset_tesis` para crear las columnas `mes_sin` y `mes_cos`.
*   **Justificación:** Resuelve el problema del "salto" numérico. Permite que el modelo entienda que diciembre (12) y enero (1) son meses contiguos climáticamente, evitando que los trate como extremos opuestos.

### 5. Depuración de Datos (Listwise Deletion)
*   **Técnica:** Eliminación de registros incompletos.
*   **Definición:** Técnica de limpieza que consiste en descartar cualquier fila que presente al menos un valor nulo (`NaN`) en sus variables predictoras.
*   **Dónde se usó:** Mediante la función `df.dropna()` en la **Fase de Depuración**.
*   **Justificación:** Garantiza que los algoritmos (como SVM o Redes Neuronales) no reciban información corrupta o incompleta que pueda sesgar el aprendizaje del modelo.

### 6. Estandarización de Datos (StandardScaler)
*   **Técnica:** Normalización por Puntuación Z (Z-score).
*   **Definición:** Transformación de variables numéricas para que tengan una media de 0 y una desviación estándar de 1.
*   **Dónde se usó:** Dentro del objeto `ColumnTransformer` bajo el nombre `num`.
*   **Justificación:** Esencial para modelos como KNN y SVM. Evita que variables con rangos grandes (como la Elevación, 0-1000m) dominen injustamente sobre variables con rangos pequeños (como el NDVI, 0-1).

### 7. Codificación One-Hot (One-Hot Encoding)
*   **Técnica:** Codificación binaria de variables categóricas.
*   **Definición:** Conversión de una columna categórica con $N$ categorías en $N$ columnas binarias independientes (ceros y unos).
*   **Dónde se usó:** Dentro del `ColumnTransformer` para la variable `cobertura_suelo`.
*   **Justificación:** Evita el "sesgo ordinal". Impide que el modelo crea que un código de suelo 80 es "superior" o "mejor" que uno de 10, tratándolos como etiquetas únicas.

### 8. Validación Temporal (Out-of-Sample Split)
*   **Técnica:** Partición estratificada por tiempo.
*   **Definición:** Separación del conjunto de datos en entrenamiento (2018-2022) y prueba (2023) basándose exclusivamente en el factor cronológico.
*   **Dónde se usó:** En la **Fase de Modelado con Validación Temporal**.
*   **Justificación:** Es la forma más rigurosa de evaluar un modelo de riesgo. Prueba si el sistema es capaz de predecir incendios en un año futuro, evitando la "autocorrelación espacial" y garantizando que el modelo realmente aprendió patrones climáticos generales.

### 9. Muestreo Puntual Geoespacial (Point Sampling)
*   **Técnica:** Extracción de valores de píxel mediante geometrías puntuales.
*   **Dónde se usó:** En GEE, usando la función `.reduceRegion()` con una escala (`scale`) definida para cada sensor (30m para NASADEM, 10m para WorldCover).
*   **Justificación:** Es el proceso que convierte un mapa (ráster) en una fila de datos (tabla). Permite asignar a cada coordenada de incendio la información exacta del terreno en ese punto.

### 10. Agregación Temporal de Datos (Temporal Aggregation)
*   **Técnica:** Reducción de datos de alta frecuencia (horarios) a niveles diarios.
*   **Dónde se usó:** En el script de clima (ERA5). Usaste `.max()` para la temperatura máxima diaria y `.sum()` para la precipitación total del día.
*   **Justificación:** El modelo de riesgo de incendios trabaja con una resolución diaria. Esta técnica permite resumir el comportamiento de las 24 horas del día en un solo valor representativo (el momento de más calor o el total de lluvia).

### 11. Generación de Variables Rezagadas (Lagged Features)
*   **Técnica:** Acumulación temporal de eventos previos.
*   **Dónde se usó:** En el cálculo de `precip_7dias_mm`. Usaste `.filterDate(fecha.advance(-7, 'day'), fecha)`.
*   **Justificación:** En incendios forestales, lo que pasó antes importa tanto como lo que pasa hoy. Esta técnica permite que el modelo entienda si el suelo viene de una sequía prolongada de una semana, lo cual eleva drásticamente el riesgo.

### 12. Composición de Imágenes (Statistical Reduction)
*   **Técnica:** Reducción por mediana espacial/temporal.
*   **Dónde se usó:** En el script de NDVI (Vegetación). Usaste `.median()` sobre una colección de imágenes Landsat/MODIS.
*   **Justificación:** Sirve para **eliminar nubes**. Al tomar la mediana de un mes, descartas los valores extremos (nubes blancas o sombras negras) y te quedas con el valor real de la salud de la vegetación en Cordillera.

### 13. Integración de Datos Multifuente (Inner Join Merging)
*   **Técnica:** Fusión de datasets basada en claves múltiples.
*   **Dónde se usó:** En el script de Pandas usando `pd.merge(..., on=['clase', 'fecha', 'longitude', 'latitude'], how='inner')`.
*   **Justificación:** Es la "columna vertebral" del dataset. Consolida 5 archivos distintos (Fuegos, Clima, Topografía, NDVI, Suelo) en una sola tabla maestra, asegurando que cada fila contenga todas las dimensiones del fenómeno.

### 14. Recorte de Valores Atípicos (Outlier Clipping)
*   **Técnica:** Limitación de rango superior e inferior.
*   **Dónde se usó:** En el cálculo de humedad: `df_clima['humedad_min_pct'].clip(upper=100)`.
*   **Justificación:** Evita errores físicos. Debido a la aproximación de la fórmula de Magnus, la humedad podría dar 101% teóricamente; el "clipping" la devuelve a un rango real (100%) para no confundir al modelo.

### 15. Auditoría de Redundancia (De-duplication)
*   **Técnica:** Eliminación de registros duplicados espaciotemporales.
*   **Dónde se usó:** En el script de Auditoría usando `df.duplicated()`.
*   **Justificación:** Evita el sobreajuste (*overfitting*). Si un mismo incendio aparece dos veces, el modelo le daría doble importancia a ese evento, sesgando los resultados del ranking TOPSIS.

### 16. Garantía de Reproducibilidad (Seed Setting)
*   **Técnica:** Fijación de semilla aleatoria.
*   **Dónde se usó:** `seed = 42`, `np.random.seed(seed)`, `random_state=seed`.
*   **Justificación:** Es vital para una tesis. Asegura que si otra persona (o el jurado) corre tus scripts, obtendrá **exactamente el mismo ranking TOPSIS** y las mismas métricas que tú presentaste en tu documento.

---

### Resumen para tu sección de "Metodología":
Para tu tesis, puedes agrupar todo esto bajo el título: **"Flujo de Ingeniería de Datos y Armonización Multifuente"**. 

Menciona que aplicaste:
1.  **Técnicas Geoespaciales:** Muestreo puntual y composición de mediana para limpieza de nubes.
2.  **Técnicas Físico-Meteorológicas:** Derivación de magnitud de viento y humedad por Magnus.
3.  **Técnicas de Machine Learning:** Estandarización, Codificación One-Hot y Transformación Cíclica.
4.  **Técnicas de Auditoría:** Limpieza de nulos, control de duplicados y validación temporal.

### 17. Derivación de Variables Topográficas (Terrain Analysis)
*   **Técnica:** Generación de productos derivados de un Modelo Digital de Elevación (DEM).
*   **Dónde se usó:** En el script de GEE: `ee.Terrain.slope(dem)` y `ee.Terrain.aspect(dem)`.
*   **Justificación:** El modelo no solo usa la altura (`elevation`), sino cómo cambia el terreno. La **pendiente** influye en la velocidad del fuego y la **orientación** influye en la cantidad de radiación solar que recibe la vegetación (secado del combustible).

### 18. Normalización Vectorial (TOPSIS Specific)
*   **Técnica:** Escalado por Norma Euclídea.
*   **Dónde se usó:** En la fase de TOPSIS: `X_norm = X_matriz / np.sqrt((X_matriz**2).sum())`.
*   **Justificación:** Es una forma de normalización diferente a la de Machine Learning. Aquí se usa para que todas las métricas (Accuracy, Tiempo, Memoria) sean comparables en un espacio geométrico, permitiendo calcular distancias a la "solución ideal".

### 19. Transformación Logarítmica de Variables de Costo
*   **Técnica:** Suavizado de distribución mediante logaritmo (`log1p`).
*   **Dónde se usó:** En el análisis TOPSIS: `X_matriz['Tiempo_Inferencia'] = np.log1p(...)`.
*   **Justificación:** El tiempo de inferencia y la memoria pueden tener valores muy dispares (outliers). Aplicar el logaritmo "comprime" estas diferencias para que un modelo muy pesado no arruine todo el ranking, permitiendo una comparación más justa.

### 20. Método del Vector Propio (AHP Weights)
*   **Técnica:** Cálculo de autovalores y autovectores.
*   **Dónde se usó:** En la parte de AHP: `eigenvalues, eigenvectors = np.linalg.eig(A)`.
*   **Justificación:** Es la técnica matemática más robusta para obtener los pesos de importancia de los criterios. No es un promedio simple; es una técnica de álgebra lineal que garantiza que los pesos reflejen fielmente tus prioridades (como darle más peso al Recall).

### 21. Validación de Consistencia (Consistency Ratio)
*   **Técnica:** Auditoría de juicios de valor.
*   **Dónde se usó:** Cálculo del `CR` en el bloque AHP.
*   **Justificación:** Evalúa si tus comparaciones fueron lógicas. Si dices que A > B y B > C, pero luego dices que C > A, el sistema detecta la inconsistencia. Esto le da validez académica a tus pesos.

### 22. Remodelación de Datos (Data Reshaping / Melting)
*   **Técnica:** Transformación de formato "Ancho" a "Largo" (Tidy Data).
*   **Dónde se usó:** Antes de graficar las barras: `df_resultados.melt(...)`.
*   **Justificación:** Es necesario para que las librerías de visualización (Seaborn) puedan agrupar las métricas por modelo de forma legible.

### 23. Serialización de Objetos (Pickling)
*   **Técnica:** Persistencia de modelos entrenados.
*   **Dónde se usó:** `pickle.dumps(pipeline)`.
*   **Justificación:** Permite medir el peso real del modelo en memoria/disco y guardarlo para uso futuro sin tener que re-entrenar (vital para la eficiencia del sistema).

### 24. Cálculo de Índices de Vegetación (Normalized Difference)
*   **Técnica:** Aritmética de bandas espectrales.
*   **Dónde se usó:** En GEE para el NDVI: `imagen.normalizedDifference(['SR_B5', 'SR_B4'])`.
*   **Justificación:** Transforma la luz reflejada por las plantas en un valor biológico (vigor fotosintético).

---

### Conclusión del Roadmap para tu tesis:

Si juntas todo esto, tienes un total de **24 técnicas de preparación y tratamiento de datos**. 

**¿Cómo organizarlo en tu documento?**
Te sugiero dividirlo en **4 bloques** en tu capítulo de metodología:
1.  **Ingeniería de Características Geoespaciales:** (NDVI, Pendiente, Orientación, Muestreo Puntual).
2.  **Tratamiento Físico-Temporal:** (Magnitud de Viento, Humedad por Magnus, Meses Cíclicos, Precipitación Acumulada).
3.  **Preprocesamiento para Machine Learning:** (StandardScaler, One-Hot Encoding, Listwise Deletion, Temporal Split).
4.  **Matemática de la Decisión Multicriterio:** (AHP por Vector Propio, Normalización Vectorial TOPSIS, Transformación Logarítmica de Costos).
