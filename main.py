# ==============================================================================
# 0. CONFIGURACIÓN E IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import re
from datetime import datetime
from pathlib import Path
import shap 
import io
# Para datos raster
import rasterio
from rasterio.merge import merge
import rioxarray
import xarray as xr
import xrspatial as xrs

# Para modelado
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Para visualización
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import contextily as cx
from matplotlib.patches import PathPatch
import matplotlib.path as mpath

# Para operaciones espaciales eficientes
from shapely.geometry import Point

# --- FIJAR SEMILLAS ALEATORIAS PARA LA REPRODUCIBILIDAD ---
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
print("Librerías cargadas exitosamente.")

# --- Rutas a las NUEVAS carpetas de datos ---
DATA_DIR = Path("data")
DEM_DIR = DATA_DIR / "DEM"
FIRMS_DIR = DATA_DIR / "FIRMS" 
COBERTURA_DIR = DATA_DIR / "Cobertura_del_Suelo"
NDVI_DIR = DATA_DIR / "NDVI" 

try:
    HUMEDAD_CSV_PATH = list((DATA_DIR / "Humedad").glob("*.csv"))[0]
    PRECIPITATION_CSV_PATH = list((DATA_DIR / "Precipitation").glob("*.csv"))[0]
    TEMPERATURE_CSV_PATH = list((DATA_DIR / "Temperature").glob("*.csv"))[0]
    WIND_CSV_PATH = list((DATA_DIR / "Wind").glob("*.csv"))[0]
    print("Rutas a los archivos CSV climáticos encontradas exitosamente.")
except IndexError:
    print("ERROR CRÍTICO: No se encontró un archivo CSV en una de las carpetas climáticas. Revisa las carpetas.")
    exit() 

MERGED_DEM_PATH = DEM_DIR / "DEM_Cordillera.tif"
DEPARTAMENTO_SHP_PATH = DATA_DIR / "COORDILLERA" / "Departamento_Coordillera.shp"
VIAS_SHP_PATH = DATA_DIR / "COORDILLERA" / "Vias_principales_Coordillera.shp"
CIUDADES_SHP_PATH = DATA_DIR / "COORDILLERA" / "Ciudades_Coordillera.shp"
DISTRITOS_SHP_PATH = DATA_DIR / "COORDILLERA" / "Distritos_Coordillera.shp"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path("outputs") / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Los resultados se guardarán en: {OUTPUT_DIR}")

# ==============================================================================
# 1. FUNCIONES AUXILIARES
# ==============================================================================

def load_firms_data(firms_folder, study_area_geom, crs):
    print("Cargando datos de focos de calor (FIRMS)...")
    firms_files = list(firms_folder.glob("**/*.csv"))
    if not firms_files: raise FileNotFoundError("No se encontraron archivos .csv en la carpeta FIRMS.")
    firms_df = pd.concat([pd.read_csv(f) for f in firms_files if 'focos_cordillera' not in f.name], ignore_index=True)
    firms_df['date'] = pd.to_datetime(firms_df['acq_date'])
    gdf_incendios = gpd.GeoDataFrame(firms_df, geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude), crs="EPSG:4326").to_crs(crs)
    gdf_incendios = gdf_incendios[gdf_incendios.within(study_area_geom)]
    gdf_incendios['fire'] = 1
    print(f"Se cargaron y filtraron {len(gdf_incendios)} puntos de incendios.")
    return gdf_incendios[['date', 'geometry', 'fire']]

def load_climate_data(temp_path, precip_path, wind_path, humidity_path):
    print("Cargando y combinando datos climáticos diarios...")
    temp_df = pd.read_csv(temp_path)[['datetime', 'temp_C']].rename(columns={'temp_C': 'temperature'})
    precip_df = pd.read_csv(precip_path)[['datetime', 'precip_mm']].rename(columns={'precip_mm': 'precipitation'})
    wind_df = pd.read_csv(wind_path)[['datetime', 'wind_speed_10m']].rename(columns={'wind_speed_10m': 'wind_speed'})
    hum_df = pd.read_csv(humidity_path)[['date', 'mean_RH']].rename(columns={'mean_RH': 'humidity'})
    
    for df in [temp_df, precip_df, wind_df, hum_df]:
        df['datetime'] = pd.to_datetime(df.get('datetime', df.get('date')))
    
    climate_dfs = [temp_df, precip_df, wind_df]
    weather_df = climate_dfs[0]
    for df_to_merge in climate_dfs[1:]: weather_df = pd.merge(weather_df, df_to_merge, on='datetime', how='outer')
    
    weather_df = weather_df.set_index('datetime')
    daily_weather = weather_df.resample('D').agg({'temperature': 'mean', 'precipitation': 'sum', 'wind_speed': 'mean'}).reset_index()
    final_weather_df = pd.merge(daily_weather, hum_df, on='datetime', how='outer').set_index('datetime').sort_index().ffill().bfill()
    print("Datos climáticos diarios cargados y combinados exitosamente.")
    return final_weather_df

def asignar_variables(df_puntos, dem_path, cobertura_folder, ndvi_mensual_folder, weather_df, vias_shp_path, ciudades_shp_path):
    """
    Asigna TODAS las variables predictoras
    """
    print("\n--- INICIANDO FASE DE ASIGNACIÓN DE VARIABLES ---")
    if df_puntos.empty: return df_puntos
    
    dataset = df_puntos.copy()
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    
    # --- 3.1. Variables Topográficas ---
    print("Asignando variables topográficas...")
    # Abrimos el DEM original en su CRS geográfico (EPSG:4674)
    dem_geo = rioxarray.open_rasterio(dem_path).squeeze()
    
    # Extraemos la elevación del DEM original, ya que nuestros puntos están en ese CRS
    x_coords = xr.DataArray(dataset.geometry.x, dims="points")
    y_coords = xr.DataArray(dataset.geometry.y, dims="points")
    dataset['elevacion'] = dem_geo.sel(x=x_coords, y=y_coords, method="nearest").values
    
    # --- CORRECCIÓN CLAVE ---
    # Reproyectamos el DEM a un CRS en metros (UTM) SOLO para calcular pendiente y orientación
    print("Reproyectando DEM para cálculo de pendiente/orientación...")
    dem_utm = dem_geo.rio.reproject("EPSG:31981")
    
    # Calculamos pendiente y orientación sobre el DEM en metros
    slope_utm = xrs.slope(dem_utm)
    aspect_utm = xrs.aspect(dem_utm)
    
    # Para extraer los valores, necesitamos las coordenadas de nuestros puntos en el nuevo CRS (UTM)
    dataset_utm = dataset.to_crs("EPSG:31981")
    x_coords_utm = xr.DataArray(dataset_utm.geometry.x, dims="points")
    y_coords_utm = xr.DataArray(dataset_utm.geometry.y, dims="points")
    
    # Extraemos los valores de pendiente y orientación
    dataset['pendiente'] = slope_utm.sel(x=x_coords_utm, y=y_coords_utm, method="nearest").values
    dataset['orientacion'] = aspect_utm.sel(x=x_coords_utm, y=y_coords_utm, method="nearest").values

    # El resto del código de esta sección no cambia
    bins = [-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 361]; labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    dataset['orientacion_cat'] = pd.cut(dataset['orientacion'], bins=bins, labels=labels, right=False, ordered=False).cat.add_categories("Plano").fillna("Plano")

    # --- El resto de la función (Cobertura, NDVI, Clima, Proximidad) permanece idéntico ---
    print("Asignando Cobertura del Suelo por año...")
    dataset['cobertura'] = np.nan
    for year in sorted(dataset['year'].unique()):
        cobertura_path = cobertura_folder / f"Cobertura_Cordillera_{year}.tif"
        if cobertura_path.exists():
            cobertura_raster = rioxarray.open_rasterio(cobertura_path, masked=True).squeeze(drop=True)
            idx_year = dataset['year'] == year
            if idx_year.any():
                group_year_proj = dataset.loc[idx_year].to_crs(cobertura_raster.rio.crs)
                x_coords_year = xr.DataArray(group_year_proj.geometry.x, dims="points")
                y_coords_year = xr.DataArray(group_year_proj.geometry.y, dims="points")
                dataset.loc[idx_year, 'cobertura'] = cobertura_raster.sel(x=x_coords_year, y=y_coords_year, method="nearest").values

    print("Asignando NDVI desde archivos mensuales...")
    dataset['ndvi'] = np.nan
    NDVI_SCALE_FACTOR = 0.0001
    ndvi_files_map = { (int(f.stem.split('_')[2]), int(f.stem.split('_')[3])): f 
                       for f in ndvi_mensual_folder.glob("NDVI_MODIS_*.tif") }
    
    for (year, month), group in dataset.groupby(['year', 'month']):
        if (year, month) in ndvi_files_map:
            raster_path = ndvi_files_map[(year, month)]
            raster = rioxarray.open_rasterio(raster_path, masked=True).squeeze(drop=True)
            group_proj = group.to_crs(raster.rio.crs)
            x_coords_group = xr.DataArray(group_proj.geometry.x, dims="points")
            y_coords_group = xr.DataArray(group_proj.geometry.y, dims="points")
            raw_values = raster.sel(x=x_coords_group, y=y_coords_group, method="nearest").values
            scaled_values = np.where(~np.isnan(raw_values), raw_values * NDVI_SCALE_FACTOR, np.nan)
            dataset.loc[group.index, 'ndvi'] = scaled_values

    print("Asignando variables meteorológicas...")
    dataset = dataset.sort_values('date')
    dataset = pd.merge_asof(dataset, weather_df, left_on='date', right_index=True, direction='backward')
    
    print("Añadiendo características de proximidad...")
    projected_crs = "EPSG:31981"
    dataset_proj = dataset.to_crs(projected_crs)
    vias_gdf = gpd.read_file(VIAS_SHP_PATH).to_crs(projected_crs)
    ciudades_gdf = gpd.read_file(CIUDADES_SHP_PATH).to_crs(projected_crs)
    dataset['dist_vias'] = dataset_proj.geometry.distance(vias_gdf.union_all())
    dataset['dist_ciudades'] = dataset_proj.geometry.distance(ciudades_gdf.union_all())

    print("Asignación de variables completada.")
    return dataset
# ==============================================================================
# 2. GENERACIÓN DE LA MUESTRA Y ASIGNACIÓN DE DATOS
# ==============================================================================
print("\n--- INICIANDO FASE DE PREPARACIÓN DE DATOS ---")
area_estudio_gdf = gpd.read_file(DEPARTAMENTO_SHP_PATH)
CRS_REFERENCE = area_estudio_gdf.crs.to_string()
study_area_bounds = area_estudio_gdf.total_bounds
study_area_polygon = area_estudio_gdf.union_all()
print("\nCreando máscara de zonas inflamables a partir de la Cobertura del Suelo...")

# Define la ruta a solo UNO de tus archivos de cobertura, el más reciente servirá.
# Necesitamos la estructura, no los valores anuales.
COBERTURA_PATH_BASE = COBERTURA_DIR / "Cobertura_Cordillera_2023.tif" # O el año que sea

# Define las clases de cobertura que consideras "inflamables"
CLASES_INFLAMABLES = [3.0, 6.0, 11.0, 12.0, 15.0]  # <-- ¡¡¡AJUSTA ESTOS VALORES!!! Son los que identificaste en QGIS.

# Carga el raster
with rasterio.open(COBERTURA_PATH_BASE) as src:
    # Lee el raster como un array de numpy
    cobertura_array = src.read(1)
    transform = src.transform
    crs = src.crs

    # Crea una máscara booleana: True donde la clase es inflamable
    mask = np.isin(cobertura_array, CLASES_INFLAMABLES)

    # Vectoriza el raster: convierte los píxeles True en polígonos
    # Esto crea una lista de (geometría, valor) para cada polígono de píxeles conectados
    shapes = list(rasterio.features.shapes(mask.astype('uint8'), mask=mask, transform=transform))

shapes_geojson = [{'geometry': geom, 'properties': {'value': val}} for geom, val in shapes]
# 2. Ahora creamos el GeoDataFrame desde esta lista bien formateada
zonas_inflamables_gdf = gpd.GeoDataFrame.from_features(shapes_geojson, crs=crs)

# 3. La disolución funciona igual que antes
zonas_inflamables_poligono = zonas_inflamables_gdf.unary_union

print("Máscara de zonas inflamables creada exitosamente.")
print(f"Área de estudio definida. CRS: {CRS_REFERENCE}")

puntos_positivos = load_firms_data(FIRMS_DIR, study_area_polygon, CRS_REFERENCE)

print("\nGenerando muestras negativas (no-incendios) con muestreo de fondo inteligente...")
n_negativos = len(puntos_positivos) if len(puntos_positivos) > 0 else 1000
puntos_negativos_list = []
minx, miny, maxx, maxy = study_area_bounds
start_date, end_date = puntos_positivos['date'].min(), puntos_positivos['date'].max()
dias_rango = (end_date - start_date).days
projected_crs = "EPSG:31981"
puntos_positivos_proj = puntos_positivos.to_crs(projected_crs)
incendios_por_dia_proj = {d: g[['geometry']] for d, g in puntos_positivos_proj.groupby(puntos_positivos_proj['date'].dt.date)}
intentos = 0
max_intentos = n_negativos * 500 # Aumentamos los intentos porque será más difícil encontrar puntos

while len(puntos_negativos_list) < n_negativos:
    intentos += 1
    if intentos > max_intentos:
        print(f"\nADVERTENCIA: Se superó el número máximo de intentos ({max_intentos}). Se generaron {len(puntos_negativos_list)} puntos.")
        break
    
    # 1. Generar un punto aleatorio dentro de los límites del departamento
    random_point_geo = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
    
    # 2. PRIMER FILTRO: ¿Está el punto dentro del área de estudio Y dentro de una zona inflamable?
    #    Usamos 'within' que es más rápido que la comprobación de cobertura del raster.
    if not random_point_geo.within(zonas_inflamables_poligono):
        continue  # Si no está en una zona inflamable, se descarta y se prueba de nuevo.

    # 3. Si pasa el primer filtro, se procede a la comprobación de proximidad a incendios
    random_date = start_date + pd.to_timedelta(np.random.randint(0, dias_rango + 1), unit='d')
    es_valido = True
    ventana_temporal_dias = 3
    distancia_min_metros = 15000

    # Proyectar el punto aleatorio a UTM para el cálculo de distancia
    # Hacemos esto solo para los puntos que ya son candidatos, es más eficiente.
    random_point_gds_proj = gpd.GeoSeries([random_point_geo], crs=CRS_REFERENCE).to_crs(projected_crs)

    for i in range(-ventana_temporal_dias, ventana_temporal_dias + 1):
        check_date = (random_date + pd.to_timedelta(i, unit='d')).date()
        if check_date in incendios_por_dia_proj:
            if incendios_por_dia_proj[check_date].distance(random_point_gds_proj.iloc[0]).min() < distancia_min_metros:
                es_valido = False
                break
    
    # 4. Si el punto es válido en ambos filtros, se añade a la lista
    if es_valido:
        puntos_negativos_list.append({'date': random_date, 'geometry': random_point_geo, 'fire': 0})
        if len(puntos_negativos_list) % 200 == 0:
            print(f"  ... {len(puntos_negativos_list)}/{n_negativos} puntos negativos generados. (Tasa de éxito: {len(puntos_negativos_list)/intentos:.2%})")
if not puntos_negativos_list:
    print("\nERROR CRÍTICO: No se pudo generar ningún punto negativo.")
    exit()

puntos_negativos = gpd.GeoDataFrame(puntos_negativos_list, crs=CRS_REFERENCE)
sample = pd.concat([puntos_positivos, puntos_negativos], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
print(f"\nMuestra final generada con {len(sample)} observaciones.")

weather_data = load_climate_data(TEMPERATURE_CSV_PATH, PRECIPITATION_CSV_PATH, WIND_CSV_PATH, HUMEDAD_CSV_PATH)
full_dataset = asignar_variables(sample, MERGED_DEM_PATH, COBERTURA_DIR, NDVI_DIR, weather_data, VIAS_SHP_PATH, CIUDADES_SHP_PATH)

# ==============================================================================
# 4. DEPURACIÓN Y MODELADO
# ==============================================================================
print("\n--- INICIANDO FASE DE DEPURACIÓN ---")
print("Valores nulos por columna ANTES de depurar:\n", full_dataset.isnull().sum())
datos_depurados = full_dataset.dropna()
print(f"\nRegistros ANTES de depurar: {len(full_dataset)} | DESPUÉS de depurar: {len(datos_depurados)}")

if len(datos_depurados) < 100:
    print("\nERROR CRÍTICO: No hay suficientes datos después de la depuración.")
    exit()

columnas_modelo = ['elevacion', 'pendiente', 'orientacion_cat', 'cobertura', 'ndvi', 'temperature', 'precipitation', 'wind_speed', 'humidity', 'dist_vias', 'dist_ciudades', 'fire']
datos_modelo = datos_depurados[columnas_modelo].copy()
datos_modelo['cobertura'] = datos_modelo['cobertura'].astype('category')
print("\nInfo del dataset final:\n"); datos_modelo.info()
print("\nVista previa del dataset final:\n", datos_modelo.head())
buffer = io.StringIO()
datos_modelo.info(buf=buffer)
info_string = buffer.getvalue()
head_string = datos_modelo.head().to_string()
ruta_resumen_dataset = OUTPUT_DIR / "dataset_summary.txt"
with open(ruta_resumen_dataset, "w") as f:
    f.write("=========================================\n")
    f.write(" RESUMEN DEL DATASET FINAL (datos_modelo)\n")
    f.write("=========================================\n\n")
    f.write("--- DataFrame .info() ---\n")
    f.write(info_string)
    f.write("\n\n--- DataFrame .head() ---\n")
    f.write(head_string)
    f.write("\n")
print(f"Resumen del dataset guardado en: {ruta_resumen_dataset}")

# ==============================================================================
# 5. MODELADO CON VALIDACIÓN ESPACIAL
# ==============================================================================
print("\n--- INICIANDO FASE DE MODELADO CON VALIDACIÓN ESPACIAL ---")
"""
Separar datos con un BLOQUE ESPACIAL
En lugar de una división aleatoria, dividiremos el departamento geográficamente.
Usaremos la longitud como criterio: entrenaremos con los datos del 70% occidental
del departamento y probaremos con el 30% oriental.
"""
# Añadimos las coordenadas 'y' al DataFrame para poder filtrar
datos_modelo['y'] = datos_depurados.geometry.y

# Calculamos el punto de corte (el percentil 70 de las latitudes)
split_point = datos_modelo['y'].quantile(0.7)

# Creamos los conjuntos de entrenamiento y prueba usando la columna 'y'
train_df = datos_modelo[datos_modelo['y'] <= split_point]
test_df = datos_modelo[datos_modelo['y'] > split_point]

# Eliminar la columna 'y' que ya no necesitamos para el entrenamiento
train_df = train_df.drop(columns=['y'])
test_df = test_df.drop(columns=['y'])


print(f"División espacial: {len(train_df)} muestras para entrenar, {len(test_df)} para probar.")

# Comprobar que ambas clases están presentes en ambos conjuntos
if train_df['fire'].nunique() < 2 or test_df['fire'].nunique() < 2:
    print("\nADVERTENCIA: La división espacial resultó en un conjunto sin ambas clases. Prueba un punto de corte diferente (ej. 0.5) o considera una estrategia de bloques más compleja.")
else:
    X_train = train_df.drop('fire', axis=1)
    y_train = train_df['fire']
    X_test = test_df.drop('fire', axis=1)
    y_test = test_df['fire']

    # --- 5.2. Pipeline de preprocesamiento (sin cambios) ---
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
# Acá cambie
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='drop')

    # --- 5.3. Definir y crear el modelo de Stacking (sin cambios) ---
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        ('svm', CalibratedClassifierCV(LinearSVC(random_state=seed, dual='auto', max_iter=2000), cv=3)),
        ('knn', KNeighborsClassifier(n_neighbors=10, n_jobs=-1))
    ]
    meta_model = LogisticRegression(solver='liblinear')
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, n_jobs=-1)

    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', stacking_model)])

    # --- 5.4. Entrenar y Evaluar ---
    print("\nEntrenando el modelo de Stacking...")
    full_pipeline.fit(X_train, y_train)
    print("Entrenamiento completado.")

    print("\nEvaluando el modelo en el conjunto de prueba ESPACIAL...")
    y_pred = full_pipeline.predict(X_test)
    y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Métricas de Rendimiento (VALIDACIÓN ESPACIAL) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nReporte de Clasificación:\n"); print(classification_report(y_test, y_pred, target_names=['No Incendio', 'Incendio']))


# ==============================================================================
# 4.5 ANÁLISIS DE SANIDAD DE LA MUESTRA Y LA PARTICIÓN ESPACIAL
# ==============================================================================
print("\n--- INICIANDO ANÁLISIS DE SANIDAD DE LA MUESTRA ---")

# --- Hipótesis 1: ¿Son las muestras negativas "demasiado fáciles"? ---
print("\nGenerando gráficos para analizar la distribución de las muestras...")

# Gráfico 1: Distribución Espacial de la Muestra Completa
# Esto nos permite ver si los puntos negativos (azules) están distribuidos de forma similar a los positivos (rojos).
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
area_estudio_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, zorder=3)
datos_depurados[datos_depurados['fire'] == 0].plot(ax=ax, marker='o', color='blue', markersize=5, label='No Incendio', alpha=0.5)
datos_depurados[datos_depurados['fire'] == 1].plot(ax=ax, marker='x', color='red', markersize=8, label='Incendio', alpha=0.7)
ax.set_title('Distribución Espacial de la Muestra de Entrenamiento', fontsize=18)
ax.legend()
cx.add_basemap(ax, crs=area_estudio_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
plt.savefig(OUTPUT_DIR / "mapa_distribucion_muestra.png", dpi=300)
plt.show()

# Gráfico 2: Comparación de la Distribución de Variables Clave
# Comparamos las distribuciones para "Incendio" vs "No Incendio".
# Si las cajas son muy diferentes, los puntos negativos son "fáciles".
variables_a_comparar = ['elevacion', 'pendiente', 'ndvi', 'dist_vias', 'dist_ciudades']
for var in variables_a_comparar:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='fire', y=var, data=datos_modelo)
    plt.title(f'Comparación de "{var}" entre Clases', fontsize=16)
    plt.xticks([0, 1], ['No Incendio', 'Incendio'])
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f'boxplot_{var}_vs_fire.png', dpi=300)
    plt.show()


# --- Hipótesis 2: ¿Es la partición espacial representativa? ---
print("\nGenerando mapa de la partición Train/Test espacial...")

# Recreamos la división para obtener los índices y geometrías
temp_df_vis = datos_depurados.copy()
temp_df_vis['x'] = temp_df_vis.geometry.x
split_point_vis = temp_df_vis['x'].quantile(0.7)
train_indices = temp_df_vis[temp_df_vis['x'] <= split_point_vis].index
test_indices = temp_df_vis[temp_df_vis['x'] > split_point_vis].index

train_gdf_vis = datos_depurados.loc[train_indices]
test_gdf_vis = datos_depurados.loc[test_indices]

# Gráfico 3: Mapa de la Partición Espacial
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
area_estudio_gdf.plot(ax=ax, facecolor='whitesmoke', edgecolor='black', linewidth=1)
train_gdf_vis.plot(ax=ax, marker='o', color='dodgerblue', markersize=5, label=f'Train ({len(train_gdf_vis)} puntos)')
test_gdf_vis.plot(ax=ax, marker='o', color='orangered', markersize=5, label=f'Test ({len(test_gdf_vis)} puntos)')

# Dibujar la línea de división
ax.axvline(x=split_point_vis, color='black', linestyle='--', linewidth=2, label=f'Línea de Corte (Longitud={split_point_vis:.2f})')

ax.set_title('Visualización de la Partición Espacial Train/Test', fontsize=18)
ax.legend()
cx.add_basemap(ax, crs=area_estudio_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
plt.savefig(OUTPUT_DIR / "mapa_particion_espacial.png", dpi=300)
plt.show()

print("--- ANÁLISIS DE SANIDAD COMPLETADO ---")

# ==========================================c====================================
# 5.6. GUARDADO DE RESULTADOS Y VISUALIZACIÓN
# ==============================================================================
print("\n--- Guardando resultados del modelo ---")

# --- Guardar el reporte de clasificación en un archivo de texto ---
report = classification_report(y_test, y_pred, target_names=['No Incendio', 'Incendio'])
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

report_path = OUTPUT_DIR / "classification_report.txt"
with open(report_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
print(f"Reporte de clasificación guardado en: {report_path}")

# --- Crear y guardar el gráfico combinado (Matriz de Confusión y Curva ROC) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Resultados del Modelo de Stacking', fontsize=16)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title("Matriz de Confusión")
ax1.set_xlabel("Predicción")
ax1.set_ylabel("Real")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--', label='Azar')
ax2.set_title('Curva de ROC', fontweight='bold')
ax2.set_xlabel('Tasa de Falsos Positivos')
ax2.set_ylabel('Tasa de Verdaderos Positivos')
ax2.legend()
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax2.set_xticks(np.arange(0, 1.1, 0.2))
ax2.grid(True, which='both', linestyle='-', linewidth=0.5)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.0])

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el supertítulo

# Guardar la figura
plot_path = OUTPUT_DIR / "performance_plot.png"
plt.savefig(plot_path, dpi=300)
print(f"Gráfico de rendimiento guardado en: {plot_path}")

# Mostrar el gráfico en pantalla
plt.show()

# ==============================================================================
# 5.7. ANÁLISIS DE EXPLICABILIDAD 
# ==============================================================================
print("\n--- INICIANDO FASE DE EXPLICABILIDAD CON SHAP ---")

# --- Configuración para el modo de ejecución ---
MODO_BORRADOR = False

if MODO_BORRADOR:
    print(">>> EJECUTANDO EN MODO BORRADOR (RÁPIDO) CON UNA MUESTRA PEQUEÑA <<<")
    n_muestras_shap = 500
else:
    print(">>> EJECUTANDO EN MODO FINAL (LENTO) CON UNA MUESTRA MÁS GRANDE <<<")
    n_muestras_shap = min(len(X_test), 2000) # Usar hasta 2000 muestras para el plot final


# --- 5.7.1. Preparar datos y modelo ---

# Extraemos el preprocesador y el modelo RandomForest ya entrenados
preprocessor_trained = full_pipeline.named_steps['preprocessor']
rf_model_trained = full_pipeline.named_steps['classifier'].estimators_[0]

# Preprocesamos los datos de prueba
X_test_processed = preprocessor_trained.transform(X_test)

# Obtenemos los nombres de las características de forma segura
feature_names = preprocessor_trained.get_feature_names_out()

# Convertimos a DataFrame y tomamos nuestra muestra
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
X_test_sample_df = X_test_processed_df.sample(n=n_muestras_shap, random_state=seed)


# --- 5.7.2. Entrenar Modelo Proxy, Crear Explicador y Calcular Explicaciones ---

# Entrenamos un modelo proxy idéntico para asegurar compatibilidad total
print("Entrenando un modelo Random Forest 'proxy' para el análisis SHAP...")
rf_proxy_model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
rf_proxy_model.fit(preprocessor_trained.transform(X_train), y_train)
print("Modelo proxy entrenado.")

# Creamos el explicador con el modelo proxy
explainer = shap.TreeExplainer(rf_proxy_model)
print("Explicador TreeExplainer de SHAP creado.")

print(f"Calculando las explicaciones SHAP para {n_muestras_shap} muestras...")

explanation_object = explainer(X_test_sample_df)
print("Cálculo de explicaciones SHAP completado.")


# --- 5.7.3. Generar y Guardar los Gráficos de Explicabilidad ---

print("Generando gráficos de SHAP...")

# Para un clasificador binario, el objeto Explanation tiene 3 dimensiones: (muestras, características, clases)
# Queremos las explicaciones para la clase 1 ("Incendio").
# Podemos acceder a ellas con la sintaxis de slicing: explanation_object[:,:,1]

# A. Gráfico de Resumen (Beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(explanation_object[:,:,1], X_test_sample_df, plot_type="dot", show=False)
plt.title("Impacto de las Características en la Predicción (Random Forest)", fontsize=16)
plt.tight_layout()
shap_summary_path = OUTPUT_DIR / "shap_summary_plot_rf.png"
plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
print(f"Gráfico de resumen SHAP guardado en: {shap_summary_path}")
plt.show()

# B. Gráfico de Barras de Importancia Media
plt.figure(figsize=(10, 8))
shap.summary_plot(explanation_object[:,:,1], X_test_sample_df, plot_type="bar", show=False)
plt.title("Importancia Media de las Características (Random Forest)", fontsize=16)
plt.tight_layout()
shap_bar_path = OUTPUT_DIR / "shap_bar_plot_rf.png"
plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
print(f"Gráfico de barras SHAP guardado en: {shap_bar_path}")
plt.show()
# ==============================================================================
# 6. APLICACIÓN: MAPA DE RIESGO
# ==============================================================================
# ... [El resto de tu script para el mapa de riesgo permanece idéntico] ...
print("\n--- INICIANDO FASE DE APLICACIÓN: MAPA DE RIESGO ---")
cell_size = 0.005 
minx, miny, maxx, maxy = study_area_bounds
xv, yv = np.meshgrid(np.arange(minx, maxx, cell_size), np.arange(miny, maxy, cell_size))
grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xv.ravel(), yv.ravel()), crs=CRS_REFERENCE)
grid_en_area = grid_gdf[grid_gdf.within(study_area_polygon)].copy()

if grid_en_area.empty:
    print("\nADVERTENCIA: No se generaron puntos en la rejilla.")
else:
    print(f"Se generaron {len(grid_en_area)} puntos en la rejilla para la predicción.")
    fecha_prediccion = pd.to_datetime("2023-08-15")
    grid_en_area.loc[:, 'date'] = fecha_prediccion
    weather_pred = weather_data
    
    grid_con_variables_raw = asignar_variables(grid_en_area, MERGED_DEM_PATH, COBERTURA_DIR, NDVI_DIR, weather_pred, VIAS_SHP_PATH, CIUDADES_SHP_PATH)
    grid_con_variables = grid_con_variables_raw.dropna()
    
    if not grid_con_variables.empty:
        print(f"Prediciendo el riesgo para {len(grid_con_variables)} puntos válidos.")
        X_grid = grid_con_variables[X_train.columns]
        prob_incendio = full_pipeline.predict_proba(X_grid)[:, 1]
        grid_con_variables['prob_incendio'] = prob_incendio
        # >>> INICIO DEL CÓDIGO PARA EXPORTAR CSV <<<
        
        # 1. Añadir columnas de longitud y latitud para el formato CSV
        # Extraemos las coordenadas X e Y de la columna de geometría.
        grid_con_variables['longitud'] = grid_con_variables.geometry.x
        grid_con_variables['latitud'] = grid_con_variables.geometry.y
        
        # 2. Seleccionar las columnas relevantes para el frontend
        # Creamos una lista para que sea fácil añadir o quitar variables en el futuro.
        columnas_para_exportar = [
            'longitud', 
            'latitud', 
            'prob_incendio', 
            'elevacion', 
            'pendiente', 
            'ndvi', 
            'temperature', 
            'precipitation', 
            'humidity', 
            'wind_speed',
            'dist_vias', 
            'dist_ciudades',
            'cobertura',
            'orientacion_cat'
        ]
        
        # Creamos el DataFrame final con solo las columnas deseadas.
        df_para_exportar = grid_con_variables[columnas_para_exportar]
        
        # 3. Definir la ruta y guardar el archivo CSV
        # Usamos la variable OUTPUT_DIR que ya está definida para guardar en la carpeta correcta.
        ruta_csv_riesgo = OUTPUT_DIR / "predicciones_riesgo.csv"
        df_para_exportar.to_csv(ruta_csv_riesgo, index=False, float_format='%.6f')
        
        # 4. Imprimir un mensaje de confirmación en la terminal
        print(f"\nTabla de predicciones guardada en: {ruta_csv_riesgo}")
        
        # >>> FIN DEL CÓDIGO PARA EXPORTAR CSV <<<
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # --- 1. Preparar datos para la interpolación ---
        points = np.array([grid_con_variables.geometry.x, grid_con_variables.geometry.y]).T
        values = grid_con_variables['prob_incendio'].values
        
        vmin = pd.Series(values).quantile(0.05)
        vmax = pd.Series(values).quantile(0.95)

        minx, miny, maxx, maxy = area_estudio_gdf.total_bounds
        grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]

        print("Interpolando resultados para crear mapa de calor...")
        interpolated_grid = griddata(points, values, (grid_x, grid_y), method='cubic')
        interpolated_grid = np.nan_to_num(interpolated_grid, nan=vmin)

        # --- 2. Dibujar el mapa de calor interpolado ---
        im = ax.imshow(
            interpolated_grid.T, 
            extent=(minx, maxx, miny, maxy), 
            origin='lower', 
            cmap='YlOrRd', 
            alpha=0.9,     
            vmin=vmin,      
            vmax=vmax       
        )
        # --- 3. Enmascarar el mapa de calor ---
        
        union_poly = area_estudio_gdf.union_all()
        path_verts, path_codes = [], []
        def add_polygon_to_path(polygon):
            path_verts.extend(np.asarray(polygon.exterior.coords.xy).T)
            path_codes.extend([mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(polygon.exterior.coords) - 1))
            for interior in polygon.interiors:
                path_verts.extend(np.asarray(interior.coords.xy).T)
                path_codes.extend([mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(interior.coords) - 1))
        
        if union_poly.geom_type == 'Polygon': add_polygon_to_path(union_poly)
        elif union_poly.geom_type == 'MultiPolygon':
            for poly in union_poly.geoms: add_polygon_to_path(poly)
        
        clip_path = mpath.Path(path_verts, path_codes)
        patch = PathPatch(clip_path, transform=ax.transData, facecolor='none')
        im.set_clip_path(patch)

        area_estudio_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        gpd.read_file(DISTRITOS_SHP_PATH).to_crs(area_estudio_gdf.crs).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.6)
        
        # --- 5. Configuración final ---
        ax.set_title(f"Mapa de Riesgo de Incendio para Cordillera\nFecha: {fecha_prediccion.date()}", fontsize=18)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        ax.set_xlim(minx - 0.05, maxx + 0.05)
        ax.set_ylim(miny - 0.05, maxy + 0.05)

        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Probabilidad de Incendio', fontsize=12)

        risk_map_path = OUTPUT_DIR / "risk_map_heatmap_final.png"
        plt.savefig(risk_map_path, dpi=300, bbox_inches='tight')
        print(f"Mapa de calor final guardado en: {risk_map_path}")
        
        plt.show()
    else:
        print("\nERROR FINAL: No se pudo generar el mapa de riesgo. La rejilla quedó vacía después de la depuración.")
        print("Valores nulos encontrados en la rejilla antes de eliminar:\n", grid_con_variables_raw.isnull().sum())


# ==============================================================================
# 7. EXPORTAR RESULTADOS PARA EL FRONTEND
# ==============================================================================
import shutil

print("\n--- COPIANDO ARTEFACTOS PARA EL FRONTEND ---")

# Ruta a la carpeta fija que leerá el frontend
frontend_data_dir = Path("frontend") / "ouput"
frontend_data_dir.mkdir(parents=True, exist_ok=True)

# Lista de los archivos que queremos mostrar en el frontend
# La clave es el nombre del archivo original, el valor es el nuevo nombre (más simple)
files_to_copy = {
    "boxplot_dist_ciudades_vs_fire.png":"boxplot_dist_ciudades_vs_fire.png",
    "boxplot_dist_vias_vs_fire.png":"boxplot_dist_vias_vs_fire.png",
    "boxplot_elevacion_vs_fire.png":"boxplot_elevacion_vs_fire.png",
    "boxplot_ndvi_vs_fire.png":"boxplot_ndvi_vs_fire.png",
    "boxplot_pendiente_vs_fire.png":"boxplot_pendiente_vs_fire.png",
    "predicciones_riesgo.csv": "predicciones_riesgo.csv",
    "classification_report.txt": "classification_report.txt",
    "dataset_summary.txt": "dataset_summary.txt",
    "performance_plot.png": "performance_plot.png",
    "risk_map_heatmap_final.png": "mapa_riesgo.png",
    "shap_summary_plot_rf.png": "shap_summary_plot.png",
    "shap_bar_plot_rf.png": "shap_bar_plot.png",
    "mapa_distribucion_muestra.png": "mapa_distribucion_muestra.png",
    "mapa_particion_espacial.png": "mapa_particion_espacial.png"
}

# Bucle para copiar y renombrar cada archivo
copied_files = []
for original_name, new_name in files_to_copy.items():
    source_path = OUTPUT_DIR / original_name
    destination_path = frontend_data_dir / new_name
    
    if source_path.exists():
        shutil.copy(source_path, destination_path)
        copied_files.append(new_name)
    else:
        print(f"ADVERTENCIA: El archivo '{original_name}' no se encontró en los outputs y no se copió.")

print(f"\nSe copiaron {len(copied_files)} archivos a la carpeta '{frontend_data_dir}'.")
print("El frontend está listo para ser visualizado.")