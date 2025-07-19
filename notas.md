# Cambios
- [ ] Utilizar el reporte mensual para la variable de humedad (csv)
- [ ] Implementar el NDVI Mensual con TIFs de MODIS (.tif)
- [ ] Añadir la capa de Cobertura del Suelo (Land Cover). Es la variable predictora más potente
- [ ] Explicar los modelos con la librería SHAP (SHapley Additive exPlanations)
- [ ] Aunque exista buen uso de la temperatura mendiante el StandardScaler debo tratar de pasar a Celsius como una buena practica
- [ ] Documentar en LATEX en nueva carpeta docs/ las herramientas/datos/variables/modelos/..etc
- [ ] Ajuste de Hiperparámetros (Investigar sobre RandomizedSearchCV de scikit-learn. Que es más eficiente que GridSearchCV) 
- [ ] Implementar Validación Cruzada Espacial (Spatial Cross-Validation) reemplazar train_test_split por una validación cruzada basada en bloques espaciales (Opcional)
- [ ] Probablemente descomente el nombre de las ciudades del mapa de riesgo

# DATASETS
- [X] HUMEDAD
- [X] VIENTO
- [X] TEMPERATURA
- [X] PRECIPITACIÓN
- [X] COBERTURA DEL SUELO
