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