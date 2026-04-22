# Cambios aplicados en el archivo main.ipynb:
## Intensidad del viento 
*   **Magnitud del Viento:** Se transformaron los componentes vectoriales `u` (Oeste-Este) y `v` (Sur-Norte) de ERA5-Land en una única variable de **intensidad física** mediante la fórmula:
    $$viento\_velocidad = \sqrt{u^2 + v^2}$$
    *Justificación:* La velocidad absoluta es el factor crítico en la propagación y desecación de combustible.
*   **Variables Cíclicas Temporales:** El mes de la captura se transformó usando funciones trigonométricas (`seno` y `coseno`).
    *Justificación:* Permite al modelo entender que diciembre (12) y enero (1) son meses contiguos en el ciclo de sequía.
*   **Estabilización Geoespacial:** Redondeo de coordenadas a 4 decimales para asegurar la integridad de los cruces (Merges) entre datasets de distintas resoluciones (GEE, NASA, ESA).


* Validación Temporal Real: Al separar 2023 como "Test", se prueba que los modelos pueden predecir incendios en un año futuro, que es el escenario real de prevención.
* Soporte para AHP-TOPSIS: Al entrenarlos por separado y guardarlos en pipelines_entrenados, ahora se pueden extraer sus métricas (Accuracy, Recall, etc.) para armar la Matriz de Decisión que requiere TOPSIS.
* Variables Corregidas: El X_train ahora usa automáticamente las columnas que "embellecimos" antes (viento_velocidad, mes_sin, mes_cos).



>  Rank 1 es aquel que mejor balancea la capacidad de detectar fuego (Recall) con la eficiencia computacional, bajo la ponderación establecida por los juicios de experto en la matriz AHP.



La Importancia Relativa es un valor que nos indica qué tanto "peso" tuvo cada variable (clima, topografía, vegetación) en las decisiones del modelo.
Si el modelo fuera una balanza, la importancia relativa nos dice qué ingrediente inclinó más la balanza hacia el resultado "Incendio" o "No Incendio". En Machine Learning, esto también se conoce como "Feature Importance" (Importancia de Características). Es lo que permite pasar de una "Caja Negra" (un modelo que nadie entiende) a un Modelo Interpretable (donde sabemos por qué predice lo que predice).




----
análisis técnico detallado que deberías incluir en tu capítulo de "Análisis de Resultados" para el departamento de Cordillera:

![alt text](image.png)

### 1. El Factor Crítico: Humedad (23.08%)
La variable `humedad_min_pct` es la más influyente con un **23% de importancia relativa**. 
*   **Interpretación:** Esto confirma científicamente que en el departamento de Cordillera, el riesgo de incendio no depende tanto de "qué tan calor hace", sino de **qué tan seco está el aire**. 
*   **Defensa:** Cuando la humedad baja, la vegetación (combustible fino) pierde agua rápidamente, volviéndose extremadamente inflamable. Es el principal "disparador" del fuego en la región.

### 2. El Peso de la Topografía: Elevación (15.44%)
Es muy interesante que la `elevation` sea la segunda variable más importante.
*   **Interpretación:** Cordillera se caracteriza por la **Cordillera de los Altos**. El hecho de que la elevación tenga un 15% de peso indica que los focos de calor tienen una distribución espacial ligada al relieve. 
*   **Defensa:** Las zonas más altas o con cambios de elevación suelen tener vientos distintos y tipos de vegetación específicos que el modelo ha detectado como zonas de mayor riesgo.

### 3. El Efecto Acumulativo de la Lluvia (~21.8%)
Si sumas `precip_diaria_mm` (11.7%) y `precip_7dias_mm` (10.1%), las precipitaciones tienen un peso combinado de casi el **22%**.
*   **Interpretación:** El modelo no solo mira si llovió hoy, sino que le da casi la misma importancia a la lluvia acumulada de la semana pasada.
*   **Defensa:** Esto demuestra que el **estrés hídrico previo** es fundamental. Si no ha llovido en 7 días, aunque hoy caiga una llovizna leve, el riesgo sigue siendo alto porque el suelo y la vegetación siguen secos.

### 4. Viento y Temperatura (Menor peso del esperado)
El `viento_velocidad` (7.6%) y la `temp_max_c` (9.7%) tienen pesos moderados.
*   **Interpretación:** Aunque son importantes, no son tan determinantes como la humedad en Cordillera. 
*   **Defensa:** Esto sugiere que, en esta región, un día de 30°C con humedad del 20% es más peligroso que un día de 40°C con humedad del 60%.

### 5. Variables de Estacionalidad y Suelo
*   **Meses (`mes_cos` / `mes_sin`):** Suman un ~7%. Esto valida que tu ingeniería de variables cíclicas funciona; el modelo entiende que hay una "temporada de incendios" marcada en el calendario paraguayo.
*   **Cobertura de Suelo:** Los valores son bajos (0.2% a 1.4%). 
    *   **¿Por qué?** Probablemente porque en las zonas de estudio el uso de suelo es muy similar (pastizales o bosques degradados), por lo que el modelo se apoya más en las variables que **sí cambian día a día** (el clima) para predecir el riesgo.

---

### Ejemplo de texto para tu Tesis:


> "Como se observa en la Figura [X], el análisis de importancia relativa de variables —utilizando Random Forest como modelo de referencia ante la naturaleza no paramétrica del ganador SVM— permite identificar los principales condicionantes del riesgo en el departamento de Cordillera. 
> 
> Destaca la **humedad relativa mínima** como el predictor más robusto (23.08%), superando incluso a la temperatura máxima (9.71%). Este hallazgo sugiere que el estrés higroscópico de la vegetación es el factor detonante predominante en la zona. 
> 
> Asimismo, el peso significativo de la **elevación** (15.44%) subraya la influencia de la orografía de la Cordillera de los Altos en la propagación y ocurrencia de incendios. Por último, la relevancia combinada de las **precipitaciones diarias y acumuladas** (21.83%) evidencia la importancia de considerar la memoria hídrica del combustible forestal para una predicción precisa."

### Tip para la defensa:
Si el jurado te pregunta por qué `cobertura_suelo_90` (Cuerpos de agua) es la última (0.0023), respondes con seguridad: *"Es lógico y correcto, ya que en cuerpos de agua el riesgo de incendio es prácticamente nulo, por lo que la variable tiene un peso mínimo en la decisión del modelo."*


----

