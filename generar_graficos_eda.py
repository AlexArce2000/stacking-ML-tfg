import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
# --- Rutas de entrada ---
# Archivo generado por tu script de modelado principal
DATOS_DEPURADOS_PKL = "datos_depurados_para_eda.pkl"
# Shapefile del límite del departamento para el mapa de densidad
DEPARTAMENTO_SHP_PATH = "data/COORDILLERA/Departamento_Coordillera.shp" 

# --- Carpeta de Salida ---
# Se creará una carpeta con este nombre para guardar los gráficos
OUTPUT_DIR = Path("outputs_analisis_exploratorio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Los gráficos de EDA se guardarán en: {OUTPUT_DIR}")

# --- CRS Métrico ---
# Asegúrate de que sea el mismo que usas en tu script principal para consistencia
METRIC_CRS = "EPSG:32721"


# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
print(f"Cargando datos desde '{DATOS_DEPURADOS_PKL}'...")
try:
    datos_depurados = pd.read_pickle(DATOS_DEPURADOS_PKL)
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo '{DATOS_DEPURADOS_PKL}'.")
    print("Por favor, ejecuta tu script de modelado principal para generar este archivo.")
    exit()

# Separar los focos de incendio (casos positivos)
focos_incendio = datos_depurados[datos_depurados['fire'] == 1].copy()

# ==============================================================================
# GRÁFICO 1: SERIE TEMPORAL DE FOCOS DE CALOR
# Objetivo: Visualizar la estacionalidad y tendencias de los incendios.
# ==============================================================================
print("Generando Gráfico 1: Serie Temporal de Focos de Calor...")
# Asegurarnos que la columna 'date' es de tipo datetime
focos_incendio['date'] = pd.to_datetime(focos_incendio['date'])

# Agrupar los focos por mes y contarlos
focos_por_mes = focos_incendio.set_index('date').resample('M').size()

# Crear el gráfico
fig, ax = plt.subplots(figsize=(15, 7))
focos_por_mes.plot(ax=ax, linewidth=2, marker='o', linestyle='-')

# Estilo y etiquetas
ax.set_title('Número de Focos de Calor Detectados por Mes (2018-2023)', fontsize=16)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Número de Focos de Calor', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Guardar el gráfico
ruta_guardado_ts = OUTPUT_DIR / "grafico_eda_serie_temporal.png"
plt.savefig(ruta_guardado_ts, dpi=300)
print(f"Gráfico guardado en: {ruta_guardado_ts}")
plt.show()


# ==============================================================================
# GRÁFICO 2: MAPA DE DENSIDAD DE KERNEL (HEATMAP ESPACIAL)
# Objetivo: Identificar visualmente las "zonas críticas" de alta concentración.
# ==============================================================================
print("Generando Gráfico 2: Mapa de Densidad de Kernel...")
# Cargar el shapefile del departamento
departamento_gdf = gpd.read_file(DEPARTAMENTO_SHP_PATH)

# Reproyectar ambos GeoDataFrames a un CRS métrico para un análisis correcto
focos_metric = focos_incendio.to_crs(METRIC_CRS)
departamento_metric = departamento_gdf.to_crs(METRIC_CRS)

# Crear el gráfico
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Dibujar el contorno del departamento
departamento_metric.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)

# Generar y dibujar el mapa de densidad de kernel
sns.kdeplot(
    x=focos_metric.geometry.x,
    y=focos_metric.geometry.y,
    ax=ax,
    fill=True,          # Rellena las áreas de densidad
    cmap="Reds",        # Paleta de colores (rojos son buenos para calor)
    alpha=0.7,          # Transparencia
    levels=10,          # Número de contornos de densidad
    zorder=2
)

# Añadir un mapa base geográfico
cx.add_basemap(ax, crs=METRIC_CRS, source=cx.providers.CartoDB.Positron)

# Estilo y etiquetas
ax.set_title('Densidad Espacial de Focos de Calor en Cordillera (2018-2023)', fontsize=16)
ax.set_xlabel('Longitud (UTM)', fontsize=12)
ax.set_ylabel('Latitud (UTM)', fontsize=12)
ax.tick_params(axis='x', rotation=45)

# Guardar el gráfico
ruta_guardado_kde = OUTPUT_DIR / "mapa_eda_densidad_kernel.png"
plt.savefig(ruta_guardado_kde, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {ruta_guardado_kde}")
plt.show()


# ==============================================================================
# GRÁFICO 3: MATRIZ DE CORRELACIÓN DE VARIABLES PREDICTORAS
# Objetivo: Entender las relaciones lineales entre las variables.
# ==============================================================================
print("Generando Gráfico 3: Matriz de Correlación...")
# Seleccionar solo las columnas numéricas que se usan como predictoras
variables_numericas = datos_depurados.select_dtypes(include=['number']).columns.tolist()
# Excluimos la variable objetivo 'fire' si está presente
if 'fire' in variables_numericas:
    variables_numericas.remove('fire')

# Calcular la matriz de correlación
matriz_correlacion = datos_depurados[variables_numericas].corr()

# Crear el gráfico (heatmap)
plt.figure(figsize=(14, 12))
sns.heatmap(
    matriz_correlacion,
    cmap='vlag',  # Paleta divergente: rojo (positivo), azul (negativo)
    annot=False,  # Poner en True si tienes pocas variables, sino se verá muy cargado
    linewidths=.5
)

# Estilo y etiquetas
plt.title('Matriz de Correlación de Variables Predictoras Numéricas', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar el gráfico
ruta_guardado_corr = OUTPUT_DIR / "matriz_eda_correlacion.png"
plt.savefig(ruta_guardado_corr, dpi=300)
print(f"Gráfico guardado en: {ruta_guardado_corr}")
plt.show()

print("\n¡Análisis exploratorio completado!")