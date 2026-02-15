## Problemas de Armonización Temporal Multifuente en Datos Climáticos

### Descripción del Problema
En el análisis de datos climáticos, es común utilizar múltiples fuentes de datos que pueden tener diferentes resoluciones temporales. Por ejemplo, algunos datasets pueden estar disponibles a nivel diario, mientras que otros pueden estar en formato mensual o anual. Esta disparidad en la resolución temporal puede generar problemas de armonización al intentar combinar y analizar estos datos conjuntamente.

### Tabla de Armonización Temporal

| Variable | Fuente | Formato | Resolución Original | Propuesta de Armonización para ML/AHP |
| :--- | :--- | :--- | :--- | :--- |
| **Focos de Incendio** | VIIRS (FIRMS) | CSV (Puntos) | Instantánea (Evento) | **Variable Objetivo (Diaria)** |
| **Temperatura / Viento** | ERA5 | CSV | Cada 3 horas | Promedio diario o Máxima diaria |
| **Precipitación** | ERA5 | CSV | Cada 3 horas | Acumulado diario (24h) |
| **Humedad Relativa** | Estación/ERA5 | CSV | Diaria | Mantener diaria |
| **NDVI** | MODIS | Raster (TIF) | Mensual | Interpolación lineal a diaria o valor constante por mes |
| **Coberura del Suelo** | MapBiomas/etc | Raster (TIF) | Anual | Valor constante para todo el año |
| **DEM (Elevación/Pendiente)** | SRTM/Otros | Raster (TIF) | Estático | No cambia (Constante) |

### Estrategias de Armonización. Proceso de integración multiescala:
Especificaciones de como el código está resolviendo el problema. 

1. Agregación de Alta Frecuencia (ERA5)

Los datos de 3 horas son demasiado volátiles para cruzarlos directamente con un mapa mensual de vegetación. Tu código lo soluciona en esta línea: 

```python
weather_df.resample('D').agg({'temperature': 'mean', 'precipitation': 'sum', 'wind_speed': 'mean'})
```
Transforma la escala de horas a días, promediando la temperatura y velocidad del viento, y sumando la precipitación para obtener un valor diario representativo.

2. Mapeo de Baja Frecuencia (NDVI y Cobertura)

En lugar de promediar un año entero (lo cual perdería la estacionalidad), el código hace un "muestreo inteligente" basado en el tiempo:

* Para NDVI: El diccionario `ndvi_files_map = { (year, month): f ... }` asegura que si un incendio ocurrió en mayo de 2021, el modelo use exactamente el raster de mayo de 2021.
* Para Cobertura: El bucle `for year in sorted(dataset['year'].unique())`: garantiza que si el terreno cambió de bosque a pastizal entre 2018 y 2023, el modelo lo sepa año con año.

3. Sincronización de Eventos (Puntos vs. Clima)

El uso de pd.merge_asof es clave:

```python
pd.merge_asof(dataset, weather_df, left_on='date', right_index=True, direction='backward')
```

* El problema: A veces un satélite registra un incendio a las 10:00 AM y tu dato climático es un promedio del día.
* La solución: Esta función alinea el evento (el punto de fuego) con el estado atmosférico exacto de ese día (o el día anterior si se usa backward), eliminando cualquier desfase temporal.

4. Validación Temporal (Split 2022-2023)

```python
train_df = datos_modelo[datos_modelo['anio'] <= 2022]
test_df = datos_modelo[datos_modelo['anio'] == 2023]
```

Esto soluciona el problema de la "autocorrelación temporal". Al entrenar con el pasado y evaluar con el futuro (2023), demuestras que tu armonización funciona para predecir, no solo para describir lo que ya pasó.


```
"Para resolver la disparidad de resoluciones temporales entre las fuentes (clima trihorario, índices mensuales y coberturas anuales), se implementó un pipeline de armonización basado en la escala diaria como unidad mínima de análisis. Los datos meteorológicos fueron agregados mediante promedios y acumulados diarios, mientras que las variables biofísicas (NDVI y Cobertura) fueron vinculadas dinámicamente según el mes y año de ocurrencia de cada evento mediante una operación de unión asincrónica (asof-join), garantizando la coherencia cronológica en todo el dataset de entrenamiento."
```



