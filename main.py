# ==============================================================================
# 0. CONFIGURACIÓN E IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# para calibración de SVM
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Para manejo de archivos y rutas
from pathlib import Path

# Para operaciones espaciales eficientes
from shapely.geometry import Point, box

print("Librerías cargadas exitosamente.")

# --- Rutas a las carpetas de datos ---
DATA_DIR = "data"
DEM_DIR = os.path.join(DATA_DIR, "DEM")
FIRMS_DIR = os.path.join(DATA_DIR, "FIRMS")
GIOVANNI_DIR = os.path.join(DATA_DIR, "Giovanni NASA")
NDVI_DIR = os.path.join(DATA_DIR, "NDVI")
MERGED_DEM_PATH = os.path.join(DEM_DIR, "merged_dem.tif")

# --- Crear carpeta de salida para los resultados ---
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path("outputs") / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Los resultados se guardarán en: {OUTPUT_DIR}")

# ==============================================================================
# 1. FUNCIONES AUXILIARES PARA CARGAR Y PREPROCESAR DATOS
# ==============================================================================

def merge_dem_tiles(dem_folder, output_path):
    """Fusiona múltiples teselas DEM en un único archivo raster."""
    if os.path.exists(output_path):
        print(f"El archivo DEM fusionado ya existe en: {output_path}")
        return
    
    print("Fusionando teselas DEM...")
    dem_files = glob.glob(os.path.join(dem_folder, "*.tif"))
    if not dem_files:
        raise FileNotFoundError("No se encontraron archivos .tif en la carpeta DEM.")
        
    src_files_to_mosaic = [rasterio.open(fp) for fp in dem_files]
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    for src in src_files_to_mosaic:
        src.close()
    print(f"Teselas DEM fusionadas y guardadas en: {output_path}")


def load_firms_data(firms_folder, study_area_geom, crs):
    """Carga, concatena y filtra los datos de FIRMS."""
    print("Cargando datos de focos de calor (FIRMS)...")
    firms_files = glob.glob(os.path.join(firms_folder, "**", "*.csv"), recursive=True)

    if not firms_files:
        raise FileNotFoundError("No se encontraron archivos .csv en la carpeta FIRMS.")
        
    df_list = [pd.read_csv(f) for f in firms_files]
    firms_df = pd.concat(df_list, ignore_index=True)
    
    firms_df['date'] = pd.to_datetime(firms_df['acq_date'])
    
    gdf_incendios = gpd.GeoDataFrame(
        firms_df,
        geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude),
        crs="EPSG:4326"
    ).to_crs(crs)
    
    # Filtrar puntos que caen dentro de la extensión del DEM
    gdf_incendios = gdf_incendios[gdf_incendios.within(study_area_geom)]
    gdf_incendios['fire'] = 1
    
    print(f"Se cargaron y filtraron {len(gdf_incendios)} puntos de incendios.")
    return gdf_incendios[['date', 'geometry', 'fire']]


# ESTA ES LA NUEVA VERSIÓN QUE DEBES PEGAR EN TU main.py
def load_giovanni_data(giovanni_folder):
    """Carga y procesa los datos de series de tiempo de Giovanni de forma robusta."""
    print("Cargando datos meteorológicos de Giovanni NASA...")

    def parse_giovanni_file(filepath, var_name):
        """
        Parsea un archivo CSV de Giovanni, detectando el encabezado buscando la línea que contiene 'time'.
        """
        skip = 0
        header_found = False
        try:
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip().lower().startswith('time,'):
                        skip = i
                        header_found = True
                        break
            
            if not header_found:
                raise ValueError(f"No se pudo encontrar la línea de encabezado (que empieza con 'time,') en {filepath}")

            df = pd.read_csv(filepath, skiprows=skip)
            
            df = df.rename(columns={df.columns[0]: 'datetime'})
            
            df = df.rename(columns={df.columns[1]: var_name})
            
            df = df[['datetime', var_name]]
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            return df

        except Exception as e:
            print(f"Ocurrió un error al procesar el archivo {filepath}: {e}")
            return None

    precip_files = glob.glob(os.path.join(giovanni_folder, "Precipitation", "*"))
    temp_files = glob.glob(os.path.join(giovanni_folder, "Temperature", "*"))
    wind_files = glob.glob(os.path.join(giovanni_folder, "Wind", "*"))

    if not precip_files: raise FileNotFoundError("No se encontró archivo de precipitación.")
    if not temp_files: raise FileNotFoundError("No se encontró archivo de temperatura.")
    if not wind_files: raise FileNotFoundError("No se encontró archivo de viento.")

    # Se usa el primer archivo encontrado en cada carpeta
    precip_df = parse_giovanni_file(precip_files[0], 'precipitation')
    temp_df = parse_giovanni_file(temp_files[0], 'temperature')
    wind_df = parse_giovanni_file(wind_files[0], 'wind_speed')

    # Comprobar que los dataframes se leyeron correctamente
    if precip_df is None or temp_df is None or wind_df is None:
        raise ValueError("No se pudieron cargar todos los archivos meteorológicos. Revisa los errores anteriores.")
    
    # Unir los 3 dataframes por su índice de fecha
    weather_df = precip_df.join(temp_df, how='outer').join(wind_df, how='outer')
    weather_df = weather_df.sort_index()
    
    print("Datos meteorológicos cargados y combinados.")
    return weather_df
# ==============================================================================
# 2. GENERACIÓN DE LA MUESTRA
# ==============================================================================
print("\n--- INICIANDO FASE DE PREPARACIÓN DE DATOS ---")

# --- 2.1. Definir el área de estudio a partir del DEM ---
merge_dem_tiles(DEM_DIR, MERGED_DEM_PATH)
with rioxarray.open_rasterio(MERGED_DEM_PATH) as dem:
    CRS_REFERENCE = dem.rio.crs.to_string()
    study_area_bounds = dem.rio.bounds()
    study_area_polygon = box(*study_area_bounds)

print(f"Área de estudio definida por la extensión del DEM. CRS: {CRS_REFERENCE}")

# --- 2.2. Cargar datos de incendios (puntos positivos) ---
puntos_positivos = load_firms_data(FIRMS_DIR, study_area_polygon, CRS_REFERENCE)

# --- 2.3. Generar puntos negativos ---
print("\nGenerando muestras negativas (no-incendios)...")
n_negativos = len(puntos_positivos)
puntos_negativos_list = []
minx, miny, maxx, maxy = study_area_bounds

# Rango de fechas para generar fechas aleatorias
start_date = puntos_positivos['date'].min()
end_date = puntos_positivos['date'].max()
dias_rango = (end_date - start_date).days

# Optimización: Pre-filtrar incendios por día para búsquedas rápidas
incendios_por_dia = {d: g[['geometry']] for d, g in puntos_positivos.set_index('date').groupby(pd.Grouper(freq='D'))}

while len(puntos_negativos_list) < n_negativos:
    random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
    random_date = start_date + timedelta(days=np.random.randint(0, dias_rango))
    
    # Comprobación espacio-temporal
    distancia_min_km = 15
    ventana_temporal_dias = 3
    
    es_valido = True
    for i in range(-ventana_temporal_dias, ventana_temporal_dias + 1):
        check_date = (random_date + timedelta(days=i)).date()
        if check_date in incendios_por_dia:
            # Comprobar distancia a los incendios de ese día
            if incendios_por_dia[check_date].distance(random_point).min() < (distancia_min_km * 1000):
                es_valido = False
                break
    if es_valido:
        puntos_negativos_list.append({'date': random_date, 'geometry': random_point, 'fire': 0})
        if len(puntos_negativos_list) % 200 == 0:
            print(f"  ... {len(puntos_negativos_list)}/{n_negativos} puntos negativos generados.")

puntos_negativos = gpd.GeoDataFrame(puntos_negativos_list, crs=CRS_REFERENCE)

# --- 2.4. Combinar y mezclar la muestra ---
sample = pd.concat([puntos_positivos, puntos_negativos], ignore_index=True)
sample = sample.sample(frac=1).reset_index(drop=True)

print(f"\nMuestra final generada con {len(sample)} observaciones.")
print(sample['fire'].value_counts())

# ==============================================================================
# 3. ASIGNACIÓN DE VARIABLES A LA MUESTRA
# ==============================================================================

def asignar_variables(df_puntos, dem_path, ndvi_folder, weather_df):
    """Asigna variables predictoras a un GeoDataFrame de puntos con fechas."""
    print("\n--- INICIANDO FASE DE ASIGNACIÓN DE VARIABLES ---")
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Comprobar si el dataframe de entrada está vacío
    if df_puntos.empty:
        print("Advertencia: El DataFrame de entrada está vacío. Devolviendo un DataFrame vacío.")
        # Devolver un dataframe con las columnas esperadas pero sin filas
        columnas_finales = df_puntos.columns.tolist() + [
            'elevacion', 'pendiente', 'orientacion', 'orientacion_cat', 
            'ndvi', 'temperature', 'precipitation', 'wind_speed'
        ]
        return gpd.GeoDataFrame(columns=columnas_finales, crs=df_puntos.crs)
    # --- FIN DE LA CORRECCIÓN ---

    dataset = df_puntos.copy()
    coords = np.array([(g.x, g.y) for g in dataset.geometry])
    
    # --- INICIO DE LA CORRECCIÓN 2 ---
    # Comprobar si se generaron coordenadas válidas
    if coords.shape[0] == 0:
        print("Advertencia: No se pudieron extraer coordenadas válidas de las geometrías. Devolviendo DataFrame.")
        return dataset
    # --- FIN DE LA CORRECCIÓN 2 ---

    # --- 3.1. Variables Topográficas ---
    print("Asignando variables topográficas...")
    dem = rioxarray.open_rasterio(dem_path).squeeze()
    dataset['elevacion'] = dem.sel(x=xr.DataArray(coords[:, 0], dims="points"), y=xr.DataArray(coords[:, 1], dims="points"), method="nearest").values
    slope = xrs.slope(dem)
    aspect = xrs.aspect(dem)
    dataset['pendiente'] = slope.sel(x=xr.DataArray(coords[:, 0], dims="points"), y=xr.DataArray(coords[:, 1], dims="points"), method="nearest").values
    dataset['orientacion'] = aspect.sel(x=xr.DataArray(coords[:, 0], dims="points"), y=xr.DataArray(coords[:, 1], dims="points"), method="nearest").values
    
    bins_orientacion = [-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 361]
    labels_orientacion = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    
    dataset['orientacion_cat'] = pd.cut(
        dataset['orientacion'], 
        bins=bins_orientacion, 
        labels=labels_orientacion, 
        right=False,
        ordered=False
    )
    dataset['orientacion_cat'] = dataset['orientacion_cat'].cat.add_categories("Plano").fillna("Plano")
    
    # --- 3.2. Variable de Vegetación (NDVI) ---
    print("Asignando NDVI...")
    dataset['ndvi'] = np.nan
    for year in dataset['date'].dt.year.unique():
        ndvi_path = os.path.join(ndvi_folder, f"NDVI_{year}.tif")
        if os.path.exists(ndvi_path):
            ndvi_raster = rioxarray.open_rasterio(ndvi_path).squeeze(drop=True)
            if ndvi_raster.rio.crs != dataset.crs:
                ndvi_raster = ndvi_raster.rio.reproject(dataset.crs)
            
            idx_year = dataset['date'].dt.year == year
            # Comprobar que hay puntos para este año antes de proceder
            if idx_year.any():
                coords_year = np.array([(g.x, g.y) for g in dataset[idx_year].geometry])
                
                ndvi_values = ndvi_raster.sel(x=xr.DataArray(coords_year[:, 0], dims="points"), y=xr.DataArray(coords_year[:, 1], dims="points"), method="nearest").values
                dataset.loc[idx_year, 'ndvi'] = ndvi_values

    # --- 3.3. Variables Meteorológicas (Giovanni) ---
    print("Asignando variables meteorológicas...")
    dataset_sorted = dataset.sort_values('date')
    # merge_asof requiere índices ordenados
    merged_data = pd.merge_asof(
        dataset_sorted,
        weather_df,
        left_on='date',
        right_index=True,
        direction='nearest' # Encuentra el registro de 3h más cercano
    )
    dataset = merged_data.sort_index()
    
    print("Asignación de variables completada.")
    return dataset

# --- Cargar datos de clima y ejecutar la asignación ---
weather_data = load_giovanni_data(GIOVANNI_DIR)
full_dataset = asignar_variables(sample, MERGED_DEM_PATH, NDVI_DIR, weather_data)

# ==============================================================================
# 4. DEPURACIÓN Y PREPARACIÓN FINAL PARA EL MODELO
# ==============================================================================
print("\n--- INICIANDO FASE DE DEPURACIÓN ---")
print(f"Registros antes de depurar: {len(full_dataset)}")
print("Valores nulos por columna:\n", full_dataset.isnull().sum())

datos_depurados = full_dataset.dropna()
print(f"\nRegistros después de depurar: {len(datos_depurados)}")

# Seleccionar columnas finales
columnas_modelo = [
    'elevacion', 'pendiente', 'orientacion_cat', 'ndvi', 
    'temperature', 'precipitation', 'wind_speed', 'fire'
]
datos_modelo = datos_depurados[columnas_modelo].copy()
datos_modelo['fire'] = datos_modelo['fire'].astype(int)

print("\nDataset final para modelado:\n", datos_modelo.head())
print("\nInfo del dataset:\n")
datos_modelo.info()

# ==============================================================================
# 5. MODELADO: STACKING (RF + SVM + KNN)
# ==============================================================================
print("\n--- INICIANDO FASE DE MODELADO ---")

# --- 5.1. Separar datos ---
X = datos_modelo.drop('fire', axis=1)
y = datos_modelo['fire']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 5.2. Pipeline de preprocesamiento ---
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 5.3. Definir y crear el modelo de Stacking ---
# --- NUEVA DEFINICIÓN DE ESTIMADORES (MÁS RÁPIDA) ---
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    # Usamos LinearSVC envuelto en CalibratedClassifierCV para velocidad y probabilidades
    ('svm', CalibratedClassifierCV(LinearSVC(random_state=42, dual='auto'), cv=3)),
    ('knn', KNeighborsClassifier(n_neighbors=10, n_jobs=-1))
]
meta_model = LogisticRegression(solver='liblinear')

stacking_model = StackingClassifier(
    estimators=estimators, final_estimator=meta_model, cv=5, n_jobs=-1
)

# --- 5.4. Pipeline completo (preprocesamiento + modelo) ---
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_model)
])

# --- 5.5. Entrenar y Evaluar ---
print("\nEntrenando el modelo de Stacking...")
full_pipeline.fit(X_train, y_train)
print("Entrenamiento completado.")

print("\nEvaluando el modelo en el conjunto de prueba...")
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Métricas de Rendimiento ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Incendio', 'Incendio']))

# --- 5.6. Visualizaciones de resultados y guardado ---
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
ax2.set_title('Curva ROC')
ax2.set_xlabel('Tasa de Falsos Positivos')
ax2.set_ylabel('Tasa de Verdaderos Positivos')
ax2.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el supertítulo

# Guardar la figura
plot_path = OUTPUT_DIR / "performance_plot.png"
plt.savefig(plot_path, dpi=300)
print(f"Gráfico de rendimiento guardado en: {plot_path}")

# Mostrar el gráfico en pantalla
plt.show()

# ==============================================================================
# 6. APLICACIÓN: MAPA DE RIESGO
# ==============================================================================
print("\n--- INICIANDO FASE DE APLICACIÓN: MAPA DE RIESGO ---")

# --- 6.1. Crear rejilla de puntos ---
cell_size = 10000  # 10 km
minx, miny, maxx, maxy = study_area_bounds
x_coords = np.arange(minx, maxx, cell_size)
y_coords = np.arange(miny, maxy, cell_size)
xv, yv = np.meshgrid(x_coords, y_coords)
grid_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())], crs=CRS_REFERENCE)
grid_en_area = grid_gdf[grid_gdf.intersects(study_area_polygon)]
if grid_en_area.empty:
    print("\nADVERTENCIA: No se generaron puntos en la rejilla que intersecten el área de estudio.")
    print("El mapa de riesgo no se puede generar. Revisa la proyección y los límites del área.")
else:
    print(f"Se generaron {len(grid_en_area)} puntos en la rejilla para la predicción.")
# --- 6.2. Asignar fecha y variables a la rejilla ---
fecha_prediccion = pd.to_datetime("2023-08-15") # Fecha de ejemplo
grid_en_area['date'] = fecha_prediccion

grid_con_variables = asignar_variables(grid_en_area, MERGED_DEM_PATH, NDVI_DIR, weather_data).dropna()

# --- 6.3. Predecir y visualizar ---
if not grid_con_variables.empty:
    X_grid = grid_con_variables[X_train.columns]
    prob_incendio = full_pipeline.predict_proba(X_grid)[:, 1]
    grid_con_variables['prob_incendio'] = prob_incendio

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gpd.GeoSeries([study_area_polygon], crs=CRS_REFERENCE).plot(ax=ax, color='none', edgecolor='black', linewidth=1.5)
    grid_con_variables.plot(column='prob_incendio', ax=ax, legend=True,
                            cmap='YlOrRd', vmin=0, vmax=1,
                            legend_kwds={'label': "Probabilidad de Incendio", 'orientation': "horizontal"})
    ax.set_title(f"Mapa de Riesgo de Incendio Estimado para {fecha_prediccion.date()}")
    
    # Guardar el mapa de riesgo
    risk_map_path = OUTPUT_DIR / "risk_map.png"
    plt.savefig(risk_map_path, dpi=300, bbox_inches='tight')
    print(f"Mapa de riesgo guardado en: {risk_map_path}")
    
    plt.show()
else:
    print("No se pudo generar el mapa de riesgo.")