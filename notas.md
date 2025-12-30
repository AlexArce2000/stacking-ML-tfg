# Cambios
- [X] Utilizar el reporte mensual para la variable de humedad (csv)
- [X] Implementar el NDVI Mensual con TIFs de MODIS (.tif)
- [X] Añadir la capa de Cobertura del Suelo (Land Cover). Es la variable predictora más potente
- [X] Explicar los modelos con la librería SHAP (SHapley Additive exPlanations)
- [X] Aunque exista buen uso de la temperatura mendiante el StandardScaler debo tratar de pasar a Celsius como una buena practica
- [X] Documentar en LATEX en nueva carpeta docs/ las herramientas/datos/variables/modelos/..etc
- [ ] Ajuste de Hiperparámetros (Investigar sobre RandomizedSearchCV de scikit-learn. Que es más eficiente que GridSearchCV) 
- [X] Implementar Validación Cruzada Espacial (Spatial Cross-Validation) reemplazar train_test_split por una validación cruzada basada en bloques espaciales (Opcional)
- [ ] Probablemente descomente el nombre de las ciudades del mapa de riesgo
- [X] Elaborar una noteboook jupyter 

# DATASETS
- [X] HUMEDAD
- [X] VIENTO
- [X] TEMPERATURA
- [X] PRECIPITACIÓN
- [X] COBERTURA DEL SUELO


# EXPERIMENTACIONES
- [ ] Por medio del estado de arte proponer experimentos
- [ ] Variaciones de modelos
- [ ] Variaciones de variables predictoras
- [ ] Con los resultados seleccionar aquellas que irían dentro de un full paper


# Experimentaciones y análisis sugeridos por Evaluadores
- [ ] Evaluación de los modelos bases (Random Forest, SVM y K-Nearest Neighbors) para obtener una  nueva tabla de "resultados" que compare las métricas clave (Accuracy, AUC, Precisión, Recall, F1-Score) de Random Forest, SVM, KNN y el modelo de Stacking. Demostrar cuantitativamente que el modelo de ensamble (stacking) ofrece un rendimiento superior y justifica su uso, respondiendo directamente a la crítica de la Evaluación 3.
- [ ] La validación en partición Este-Oeste (ya los había hecho pero para demostrar). Validación Cruzada Espacial (Spatial Cross-Validation): Dividir el mapa en varios "bloques" o "folds" geográficos. Entrena el modelo usando algunos bloques y valida en el bloque restante, rotando hasta que todos los bloques hayan sido usados para validación. Justificar de manera sólida tu método de validación y demostrar que los resultados no son un artefacto de la división norte-sur, abordando una de las principales preocupaciones de la Evaluación 2.
- [ ] Utiliza el modelo Random Forest (que ya estás usando como modelo base) para calcular la importancia de cada una de las 11 variables predictoras. Esto te dirá qué factores (ej. temperatura, distancia a carreteras, tipo de vegetación) tienen más peso en la predicción de incendios. Un gráfico de barras en la sección de "Resultados" que muestre la importancia relativa de cada variable. Esto responde a la sugerencia de la Evaluación 2 sobre detallar el criterio de selección de variables.
- [ ] **Opcional**. Aplicar el modelo entrenado para generar un mapa de riesgo para otro departamento de Paraguay que tenga características diferentes a Cordillera.
- [ ] Validación Temporal Post-Agosto 2023, si ya se tiene los datos, utilizar el modelo ya entrenado para predecir el riesgo en ese nuevo periodo y comparar las predicciones con los incendios que realmente ocurrieron.


# Nuevo artículo como base las variables más importantes para la predicción de incendios forestales.
1. En lugar de hablar sobre el modelo predictivo, mencionar sobre las variables más importantes (sacar del mismo modelo), con qué variable logra el mayor porcentaje de aciertos...cuánto más diferente es del otro artículo, mejor...por una parte atacamos las variables más importantes y por otra parte construimos modelos predictivos con algoritmos individuales y también hicimos stacking.
2. Usar la explicación de SHAP para ver cómo cada variable afecta la predicción del modelo.


**----- agregados luego de la revisión diciembre 2025 -----**
# FILTRO DE FIRMS
Anomalías térmicas considerados como incendios forestales
- Filtrar las columnas "confidence" y "type" \
Para la columna type: Debo quedarme con el `type==0`
```
0 = Presumed vegetation fire (Incendio de vegetación probable).
1 = Active volcano (Volcán). // no aplicable
2 = Other static land source (Otras fuentes estáticas, como industrias).
3 = Offshore (En el mar/agua). // no aplicable
```
Para la columna confidence: filtar los n(nominal) y h (high) 
```
l (low): Baja confianza. Puede ser un falso positivo (reflejo del sol, suelo caliente). Mejor descartarlos.
n (nominal): Confianza media. Son útiles.
h (high): Alta confianza. Son incendios seguros.
```


# Comparación de Modelos: Ponderación de Factores
- Utilización del método híbrido Entropía-TOPSIS.
Al usarlo se deja de discutir sobre "cuál tiene el AUC más alto" (donde la diferencia es mínima) y se pasa a evaluar la calidad integral del modelo.

### TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) (Tecnica para el Ordenamiento por Similitud con la Solución Ideal)
- La mejor alternativa es la que está más cerca de la solución ideal positiva y más lejos de la solución ideal negativa.
- Método de evaluación y selección de múltiples criterios.
- TOPSIS permite ordenar los modelos según su desempeño global considerando simultáneamente diversas métricas.

Datos → Modelos → Métricas → TOPSIS → Ranking → Modelo final

# Notas
- Problemas de clasificación se utiliza Accuracy, Precision, Recall, F1-Score (¿Cuántas veces acerté la categoría?). Problemas de Regresión se utiliza RMSE, MAE, R² (¿Qué tan cerca estuve del valor real?).

```
 me gustaría que hagamos el análisis comparativo entre los modelos y  el stacking para ese análisis comparativo, me gustaría que busquen trabajos similares que hacen esos análisis entonces, para que sepamos sepamos cuál es la forma más justa de comparar esos modelos no se puede comparar directamente que se yo la precisión de uno versus el otro verdad? Tendría que haber una especie de ponderación entre cada factor a comparar, entonces en ese sentido, quiero que busquen en trabajos similares que han hecho a comparación o eso ese tipo de comparación vean vean los factores a comparar y cómo se hace la ponderación, cuál es más importante que peso, se le haga cada uno entonces para esa forma que sea mucho más científica. La la comparación en sí, entonces enfóquense en eso y después nos reunimos a finales de este año avamos a reunirnos Para dar las orientaciones para hablar sobre la orientaciones Que van a hacer en el en el verano
```
# Métricas utilizadas

## Accuracy (exactitud):
El porcentaje de predicciones correctas sobre el total de predicciones realizadas. Es una métrica general que indica qué tan bien el modelo clasifica los datos.

### Macro avg (promedio democrático):
Trata a todas las clases por igual, sin importar cuántos datos tengan. Suma la nota de "No Incendio" y la nota de "Incendio", y divide entre 2. Le da la misma importancia a la clase pequeña que a la clase grande. Cuando te importa muchísimo que el modelo funcione bien en ambas cosas, incluso si una es muy rara. Es un promedio estricto.
### Weighted avg (promedio ponderado): 
Le da más importancia a la clase que tiene más datos. 

## AUC-ROC:
Área bajo la curva ROC (Receiver Operating Characteristic). Mide la capacidad del modelo para distinguir entre clases (incendio vs no incendio) en diferentes umbrales de decisión. Un AUC cercano a 1 indica un excelente poder de discriminación.

## Precision:
(desde la predicción del modelo) La proporción de verdaderos positivos entre todas las predicciones positivas realizadas por el modelo. Indica qué tan confiables son las predicciones de incendios hechas por el modelo. Una alta precisión significa pocas falsas alarmas.

## Recall:
(desde los hechos) La proporción de verdaderos positivos entre todos los casos reales positivos. Mide la capacidad del modelo para detectar incendios reales. Un alto recall significa que el modelo captura la mayoría de los incendios, minimizando los falsos negativos.

## F1-Score:
Un promedio balanceado entre detectar incendios (Recall) y no dar falsas alarmas (Precision). Es vital en problemas desbalanceados (donde hay pocos incendios y mucha tierra segura). Un F1 alto significa que el modelo es confiable. Ayuda a desempatar. Si dos modelos tienen la misma Accuracy, el que tenga mejor F1 es el que mejor entiende el fenómeno del fuego.

## Specificity (especificidad):
La proporción de verdaderos negativos entre todos los casos reales negativos. Mide la capacidad del modelo para identificar correctamente las áreas sin incendios. Una alta especificidad significa pocas falsas alarmas en áreas seguras.

## support:
Cantidad de datos reales utilizados.

## Métricas de Eficiencia
### Tiempo de inferencia:
Es el tiempo que tarda el modelo en dar una respuesta una vez que ya está entrenado y le entregas datos nuevos. 
### Memoria MB (Memory Footprint):
Es el tamaño físico que ocupa el modelo "guardado" en la memoria RAM o en el disco duro para poder funcionar.
