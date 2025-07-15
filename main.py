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
from scipy.interpolate import griddata
import contextily as cx
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.path as mpath 
from matplotlib.patches import PathPatch
# Para manejo de archivos y rutas
from pathlib import Path

# Para operaciones espaciales eficientes
from shapely.geometry import Point, box
# --- FIJAR SEMILLAS ALEATORIAS PARA LA REPRODUCIBILIDAD ---
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
print("Librerías cargadas exitosamente.")

# --- Rutas a las carpetas de datos ---
DATA_DIR = "data"
DEM_DIR = os.path.join(DATA_DIR, "DEM")
FIRMS_DIR = os.path.join(DATA_DIR, "FIRMS")
GIOVANNI_DIR = os.path.join(DATA_DIR, "Giovanni NASA")
NDVI_DIR = os.path.join(DATA_DIR, "NDVI")
MERGED_DEM_PATH = os.path.join(DEM_DIR, "merged_dem.tif")
HUMEDAD_TIF_PATH = os.path.join(GIOVANNI_DIR, "Humedad", "Humedad_Cordillera.tif")
DEPARTAMENTO_SHP_PATH = "data/COORDILLERA/Departamento_Coordillera.shp"
VIAS_SHP_PATH = "data/COORDILLERA/Vias_principales_Coordillera.shp"
CIUDADES_SHP_PATH = "data/COORDILLERA/Ciudades_Coordillera.shp"
DISTRITOS_SHP_PATH = "data/COORDILLERA/Distritos_Coordillera.shp"

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

DEPARTAMENTO_SHP_PATH = "data/COORDILLERA/Departamento_Coordillera.shp" 

# --- 2.1. Definir el área de estudio a partir del SHAPEFILE ---
print(f"Cargando área de estudio desde: {DEPARTAMENTO_SHP_PATH}")
try:
    area_estudio_gdf = gpd.read_file(DEPARTAMENTO_SHP_PATH)
except Exception as e:
    print(f"ERROR: No se pudo leer el shapefile en la ruta: {DEPARTAMENTO_SHP_PATH}")
    print(f"Asegúrate de que la ruta es correcta y que los archivos .shp, .shx, .dbf existen.")
    exit() # Detener el script si no se puede cargar el área de estudio

# Obtener los límites, el polígono y el CRS del shapefile
CRS_REFERENCE = area_estudio_gdf.crs.to_string()
study_area_bounds = area_estudio_gdf.total_bounds
study_area_polygon = area_estudio_gdf.unary_union

print(f"Área de estudio definida por el polígono del Dpto. de Cordillera. CRS: {CRS_REFERENCE}")

# --- Fusionar DEM (sigue siendo necesario para los datos topográficos) ---
merge_dem_tiles(DEM_DIR, MERGED_DEM_PATH)

# --- 2.2. Cargar datos de incendios (puntos positivos) DENTRO de Cordillera ---
puntos_positivos = load_firms_data(FIRMS_DIR, study_area_polygon, CRS_REFERENCE)

# --- 2.3. Generar puntos negativos ---
print("\nGenerando muestras negativas (no-incendios)...")
n_negativos = len(puntos_positivos) if len(puntos_positivos) > 0 else 1000 # Evitar 0
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

# ==============================================================================
# 3. ASIGNACIÓN DE VARIABLES A LA MUESTRA
# ==============================================================================

def asignar_variables(df_puntos, dem_path, ndvi_folder, weather_df, humedad_tif_path, vias_shp_path, ciudades_shp_path):
    """
    Asigna TODAS las variables predictoras (topográficas, vegetación, clima, humedad y proximidad)
    a un GeoDataFrame de puntos con fechas.
    """
    print("\n--- INICIANDO FASE DE ASIGNACIÓN DE VARIABLES ---")
    
    if df_puntos.empty:
        print("Advertencia: DataFrame de entrada vacío, no se asignarán variables.")
        return df_puntos

    dataset = df_puntos.copy()
    dataset['dia_del_ano'] = dataset['date'].dt.dayofyear
    
    coords = np.array([(g.x, g.y) for g in dataset.geometry])
    if coords.shape[0] == 0: return dataset

    # --- 3.1. Variables Topográficas ---
    print("Asignando variables topográficas...")
    dem = rioxarray.open_rasterio(dem_path).squeeze()
    dataset['elevacion'] = dem.sel(x=xr.DataArray(coords[:, 0], dims="points"), y=xr.DataArray(coords[:, 1], dims="points"), method="nearest").values
    slope, aspect = xrs.slope(dem), xrs.aspect(dem)
    dataset['pendiente'] = slope.sel(x=xr.DataArray(coords[:, 0], dims="points"), y=xr.DataArray(coords[:, 1], dims="points"), method="nearest").values
    dataset['orientacion'] = aspect.sel(x=xr.DataArray(coords[:, 0], dims="points"), y=xr.DataArray(coords[:, 1], dims="points"), method="nearest").values
    bins = [-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 361]; labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    dataset['orientacion_cat'] = pd.cut(dataset['orientacion'], bins=bins, labels=labels, right=False, ordered=False)
    dataset['orientacion_cat'] = dataset['orientacion_cat'].cat.add_categories("Plano").fillna("Plano")

    # --- 3.2. Variable de Humedad (desde TIF) ---
    print("Asignando Humedad Relativa...")
    if os.path.exists(humedad_tif_path):
        try:
            humedad_raster = rioxarray.open_rasterio(humedad_tif_path).squeeze(drop=True).rio.reproject(dataset.crs)
            dataset['humedad'] = humedad_raster.sel(x=xr.DataArray(coords[:, 0], dims="points"), y=xr.DataArray(coords[:, 1], dims="points"), method="nearest").values
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo procesar el archivo de Humedad. Error: {e}. Asignando -1.")
            dataset['humedad'] = -1
    else:
        print(f"ADVERTENCIA: No se encontró el archivo de Humedad en {humedad_tif_path}. Asignando -1.")
        dataset['humedad'] = -1

    # --- 3.3. Variable de Vegetación (NDVI) ---
    print("Asignando NDVI...")
    # ... (la lógica de NDVI sigue igual) ...
    dataset['ndvi'] = np.nan
    for year in dataset['date'].dt.year.unique():
        ndvi_path = os.path.join(ndvi_folder, f"NDVI_{year}.tif")
        if os.path.exists(ndvi_path):
            ndvi_raster = rioxarray.open_rasterio(ndvi_path).squeeze(drop=True).rio.reproject(dataset.crs)
            idx_year = dataset['date'].dt.year == year
            if idx_year.any():
                coords_year = np.array([(g.x, g.y) for g in dataset[idx_year].geometry])
                dataset.loc[idx_year, 'ndvi'] = ndvi_raster.sel(x=xr.DataArray(coords_year[:, 0], dims="points"), y=xr.DataArray(coords_year[:, 1], dims="points"), method="nearest").values

    # --- 3.4. Variables Meteorológicas (Giovanni) ---
    print("Asignando variables meteorológicas...")
    dataset = pd.merge_asof(dataset.sort_values('date'), weather_df, left_on='date', right_index=True, direction='nearest').sort_index()
    
    # --- 3.5. Características de Proximidad ---
    print("Añadiendo características de proximidad...")
    if os.path.exists(vias_shp_path):
        vias_gdf = gpd.read_file(vias_shp_path).to_crs(dataset.crs)
        dataset['dist_vias'] = dataset.geometry.distance(vias_gdf.union_all())
    else:
        dataset['dist_vias'] = -1

    if os.path.exists(ciudades_shp_path):
        ciudades_gdf = gpd.read_file(ciudades_shp_path).to_crs(dataset.crs)
        dataset['dist_ciudades'] = dataset.geometry.distance(ciudades_gdf.union_all())
    else:
        dataset['dist_ciudades'] = -1

    print("Asignación de variables completada.")
    return dataset

# --- Cargar datos de clima y ejecutar la asignación ---
weather_data = load_giovanni_data(GIOVANNI_DIR)
full_dataset = asignar_variables(sample, MERGED_DEM_PATH, NDVI_DIR, weather_data, HUMEDAD_TIF_PATH, VIAS_SHP_PATH, CIUDADES_SHP_PATH)
# ==============================================================================
# 4. DEPURACIÓN Y PREPARACIÓN FINAL PARA EL MODELO
# ==============================================================================
print("\n--- INICIANDO FASE DE DEPURACIÓN ---")
print(f"Registros antes de depurar: {len(full_dataset)}")
print("Valores nulos por columna:\n", full_dataset.isnull().sum())

datos_depurados = full_dataset.dropna()
print(f"\nRegistros después de depurar: {len(datos_depurados)}")

# Seleccionar columnas finales
columnas_base = [
    'elevacion', 'pendiente', 'orientacion_cat','humedad', 'ndvi', 
    'temperature', 'precipitation', 'wind_speed'
]
columnas_modelo = columnas_base + ['fire']
# Añadir las nuevas características de proximidad SOLO SI existen
if 'dist_vias' in datos_depurados.columns and datos_depurados['dist_vias'].nunique() > 1:
    print("-> Característica 'dist_vias' encontrada y añadida.")
    columnas_modelo.insert(-1, 'dist_vias')
else:
    print("ADVERTENCIA: 'dist_vias' no se encontró o no tiene variación. No se usará.")

if 'dist_ciudades' in datos_depurados.columns and datos_depurados['dist_ciudades'].nunique() > 1:
    print("-> Característica 'dist_ciudades' encontrada y añadida.")
    columnas_modelo.insert(-1, 'dist_ciudades')
else:
    print("ADVERTENCIA: 'dist_ciudades' no se encontró o no tiene variación. No se usará.")

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
# 6. APLICACIÓN: MAPA DE RIESGO
# ==============================================================================
print("\n--- INICIANDO FASE DE APLICACIÓN: MAPA DE RIESGO ---")

# --- 6.1. Crear rejilla de puntos dentro del polígono del departamento ---
cell_size = 0.005 
minx, miny, maxx, maxy = study_area_bounds
x_coords = np.arange(minx, maxx, cell_size)
y_coords = np.arange(miny, maxy, cell_size)
xv, yv = np.meshgrid(x_coords, y_coords)
grid_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())], crs=CRS_REFERENCE)

# Filtrar los puntos para que estén estrictamente dentro del polígono del departamento
grid_en_area = grid_gdf[grid_gdf.within(study_area_polygon)]

if grid_en_area.empty:
    print("\nADVERTENCIA: No se generaron puntos en la rejilla que caigan dentro del polígono del departamento.")
    print("Prueba un 'cell_size' aún más pequeño si el departamento es muy irregular.")
else:
    print(f"Se generaron {len(grid_en_area)} puntos en la rejilla para la predicción.")
    
    # --- 6.2. Asignar fecha y variables a la rejilla ---
    fecha_prediccion = pd.to_datetime("2023-01-01")
    grid_en_area['date'] = fecha_prediccion

    grid_con_variables_raw = asignar_variables(sample, MERGED_DEM_PATH, NDVI_DIR, weather_data, HUMEDAD_TIF_PATH, VIAS_SHP_PATH, CIUDADES_SHP_PATH)
    
    # Eliminar filas que tengan algún valor nulo después de la asignación
    grid_con_variables = grid_con_variables_raw.dropna()
    
    # --- 6.3. Predecir y visualizar ---
    if not grid_con_variables.empty:
        print(f"Prediciendo el riesgo para {len(grid_con_variables)} puntos válidos.")
        X_grid = grid_con_variables[X_train.columns]
        prob_incendio = full_pipeline.predict_proba(X_grid)[:, 1]
        grid_con_variables['prob_incendio'] = prob_incendio

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
            cmap='YlOrRd', # <--- ¡CAMBIO REALIZADO AQUÍ!
            alpha=0.9,      # Puedes ajustar la transparencia si quieres
            vmin=vmin,      
            vmax=vmax       
        )
        
        # --- 3. Enmascarar el mapa de calor ---
        import matplotlib.path as mpath
        from matplotlib.patches import PathPatch
        
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

        # --- 4. Dibujar contornos ---
        area_estudio_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1) 
        if os.path.exists(DISTRITOS_SHP_PATH):
            distritos_gdf = gpd.read_file(DISTRITOS_SHP_PATH).to_crs(area_estudio_gdf.crs)
            distritos_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.6)
        # --- AÑADIR CIUDADES Y SUS NOMBRES SE VE FEO POR ESO COMENTÉ---
        # if os.path.exists(CIUDADES_SHP_PATH):
        #     ciudades_gdf = gpd.read_file(CIUDADES_SHP_PATH).to_crs(area_estudio_gdf.crs)
            
        #     # 1. Dibujar un punto para cada ciudad
        #     ciudades_gdf.plot(
        #         ax=ax, 
        #         marker='o',
        #         color="white",
        #         edgecolor="black",
        #         markersize=30
        #     )

        #     # 2. Escribir el nombre al lado de cada punto
        #     if "DIST_DESC_" in ciudades_gdf.columns:
        #         for x, y, label in zip(ciudades_gdf.geometry.x, ciudades_gdf.geometry.y, ciudades_gdf["DIST_DESC_"]):
        #             ax.text(
        #                 x + 0.005,  # Desplazamiento a la derecha
        #                 y,
        #                 label.title(), # Pone el texto en formato de título
        #                 fontsize=9,
        #                 ha='left',
        #                 va='center',
        #                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.2) # Fondo para legibilidad
        #             )
        #     else:
        #         print("ADVERTENCIA: No se encontró la columna 'DIST_DESC_' para las etiquetas de ciudades.")
        
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
        print("\nERROR FINAL: No se pudo generar el mapa de riesgo.")
        print("Causa: Después de eliminar filas con valores nulos, no quedó ningún punto en la rejilla.")
        print("Esto suele ocurrir si los puntos caen fuera de la cobertura de algún raster (ej. NDVI).")
        print("Valores nulos encontrados en la rejilla antes de eliminar:")
        print(grid_con_variables_raw.isnull().sum())