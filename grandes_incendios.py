import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPoint
import matplotlib.pyplot as plt
import contextily as cx
from datetime import timedelta, datetime
from pathlib import Path 
import os 

FIRMS_CSV_PATH = "data/FIRMS/fire_archive_J1V-C2_637085.csv"
DEPARTAMENTO_SHP_PATH = "data/COORDILLERA/Departamento_Coordillera.shp" 
# --- Parámetros para agrupar incendios ---
# Dos puntos pertenecen al mismo incendio si están a menos de X metros y Y días de diferencia.
# Puedes experimentar con estos valores.
SPATIAL_THRESHOLD_METERS = 2000  # 2 km
TEMPORAL_THRESHOLD_DAYS = 2      # 2 días

# --- Parámetros para definir un "Gran Incendio Forestal" ---
# Un evento se considera "grande" si cumple AMBOS criterios.
MIN_DETECTIONS_FOR_LARGE_FIRE = 15  # Debe tener al menos 15 detecciones de focos de calor.
MIN_AREA_HA_FOR_LARGE_FIRE = 100    # El área de su polígono envolvente debe ser > 100 hectáreas.

# Sistema de Coordenadas de Referencia (CRS) a usar para medir distancias en metros.
# EPSG:32721 es UTM Zone 21S, apropiado para Paraguay.
METRIC_CRS = "EPSG:32721"

print("Configuración cargada.")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path("outputs_grandes_incendios") / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Los resultados se guardarán en: {OUTPUT_DIR}")
# ==============================================================================
# 2. CARGAR Y PREPARAR LOS DATOS
# ==============================================================================
print("Cargando datos...")
# Cargar el límite del departamento
departamento_gdf = gpd.read_file(DEPARTAMENTO_SHP_PATH)

# Cargar los datos de focos de calor
try:
    firms_df = pd.read_csv(FIRMS_CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo en la ruta: {FIRMS_CSV_PATH}")
    exit()

# Convertir a GeoDataFrame
firms_gdf = gpd.GeoDataFrame(
    firms_df,
    geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude),
    crs="EPSG:4326"  # CRS estándar para latitud/longitud
)

# Filtrar los focos que están DENTRO del departamento de Cordillera
print("Filtrando focos de calor para el departamento de Cordillera...")
gdf_cordillera = gpd.sjoin(firms_gdf, departamento_gdf, how="inner", predicate='within')
if gdf_cordillera.empty:
    print("No se encontraron focos de calor dentro del departamento de Cordillera en el archivo proporcionado.")
    exit()

# Convertir fecha a formato datetime para poder operar con ellas
gdf_cordillera['acq_date'] = pd.to_datetime(gdf_cordillera['acq_date'])

# Reproyectar a un CRS métrico para poder medir distancias en metros
gdf_cordillera = gdf_cordillera.to_crs(METRIC_CRS)

print(f"Se encontraron {len(gdf_cordillera)} focos de calor en Cordillera.")


# ==============================================================================
# 3. ALGORITMO DE AGRUPAMIENTO (CLUSTERING) ESPACIO-TEMPORAL
# ==============================================================================
print("\nIniciando algoritmo de agrupamiento para identificar eventos de incendio...")
# Ordenamos los puntos por fecha, es una buena práctica para este tipo de algoritmo
gdf_cordillera = gdf_cordillera.sort_values(by='acq_date').reset_index(drop=True)

# Añadimos una columna para identificar a qué evento de incendio pertenece cada punto
gdf_cordillera['fire_id'] = -1
current_fire_id = 0

# Iteramos sobre cada punto para asignarlo a un clúster (evento de incendio)
for i in range(len(gdf_cordillera)):
    # Si el punto ya tiene un ID de incendio, lo saltamos
    if gdf_cordillera.loc[i, 'fire_id'] != -1:
        continue

    # Si no, este punto es el inicio de un nuevo evento de incendio
    # Creamos una "pila" de puntos por revisar para este clúster, empezando con el actual
    points_to_check_stack = [i]
    gdf_cordillera.loc[i, 'fire_id'] = current_fire_id

    while len(points_to_check_stack) > 0:
        # Sacamos un punto de la pila
        current_point_idx = points_to_check_stack.pop()
        current_point_geom = gdf_cordillera.loc[current_point_idx, 'geometry']
        current_point_date = gdf_cordillera.loc[current_point_idx, 'acq_date']

        # Buscamos vecinos en el espacio y en el tiempo que AÚN no hayan sido asignados a un incendio
        spatial_neighbors = gdf_cordillera[gdf_cordillera.geometry.distance(current_point_geom) <= SPATIAL_THRESHOLD_METERS]
        
        # Filtramos por tiempo y por los que no tienen ID
        time_neighbors_mask = (spatial_neighbors['acq_date'] - current_point_date).abs() <= timedelta(days=TEMPORAL_THRESHOLD_DAYS)
        unassigned_neighbors_mask = spatial_neighbors['fire_id'] == -1
        
        final_neighbors = spatial_neighbors[time_neighbors_mask & unassigned_neighbors_mask]

        # Para cada vecino encontrado, lo asignamos a este incendio y lo añadimos a la pila para buscar SUS vecinos
        for neighbor_idx in final_neighbors.index:
            gdf_cordillera.loc[neighbor_idx, 'fire_id'] = current_fire_id
            points_to_check_stack.append(neighbor_idx)

    # Cuando la pila se vacía, significa que hemos encontrado todos los puntos de este evento.
    # Incrementamos el ID para el siguiente evento que encontremos.
    current_fire_id += 1

print(f"Agrupamiento completado. Se identificaron {current_fire_id} eventos de incendio potenciales.")

# ==============================================================================
# 4. ANÁLISIS DE LOS EVENTOS DE INCENDIO Y FILTRADO DE LOS GRANDES
# ==============================================================================
print("\nAnalizando eventos y filtrando los incendios grandes...")
fire_events = []
# Agrupamos por el ID de incendio que acabamos de asignar
for fire_id, cluster_gdf in gdf_cordillera.groupby('fire_id'):
    if len(cluster_gdf) < 2: continue # Ignoramos eventos de un solo punto

    # Calcular la "huella" del incendio con un polígono envolvente (convex hull)
    # Esto nos da una idea de la extensión máxima de las detecciones
    hull = cluster_gdf.unary_union.convex_hull
    
    # Calculamos las métricas de cada evento
    event_data = {
        'fire_id': fire_id,
        'num_detections': len(cluster_gdf),
        'start_date': cluster_gdf['acq_date'].min(),
        'end_date': cluster_gdf['acq_date'].max(),
        'duration_days': (cluster_gdf['acq_date'].max() - cluster_gdf['acq_date'].min()).days + 1,
        'avg_frp': cluster_gdf['frp'].mean(),
        'area_ha': hull.area / 10000,  # Convertir de m² a hectáreas
        'geometry': hull
    }
    fire_events.append(event_data)

if not fire_events:
    print("No se pudieron agrupar eventos de incendio con los parámetros actuales.")
    exit()

# Convertimos la lista de análisis en un GeoDataFrame
events_gdf = gpd.GeoDataFrame(fire_events, crs=METRIC_CRS)

# Filtramos para quedarnos solo con los "Grandes Incendios Forestales"
large_fires_gdf = events_gdf[
    (events_gdf['num_detections'] >= MIN_DETECTIONS_FOR_LARGE_FIRE) &
    (events_gdf['area_ha'] >= MIN_AREA_HA_FOR_LARGE_FIRE)
].sort_values(by='area_ha', ascending=False).reset_index(drop=True)


print("\n--- RESULTADOS: GRANDES INCENDIOS FORESTALES IDENTIFICADOS ---")
if large_fires_gdf.empty:
    print("No se identificaron grandes incendios forestales con los criterios definidos.")
    print(f"Criterios: Mínimo {MIN_DETECTIONS_FOR_LARGE_FIRE} detecciones y {MIN_AREA_HA_FOR_LARGE_FIRE} ha de huella.")
else:
    display_cols = ['fire_id', 'area_ha', 'num_detections', 'duration_days', 'start_date', 'end_date', 'avg_frp']
    print("Resumen de grandes incendios:")
    print(large_fires_gdf[display_cols].round(2))

    # Guardar resultados (código sin cambios)
    csv_output_path = OUTPUT_DIR / "grandes_incendios_identificados.csv"
    large_fires_gdf[display_cols].round(2).to_csv(csv_output_path, index=False)
    print(f"\nTabla de resultados guardada en: {csv_output_path}")

    geopackage_output_path = OUTPUT_DIR / "huellas_grandes_incendios.gpkg"
    large_fires_gdf.to_file(geopackage_output_path, driver='GPKG')
    print(f"Geometría de las huellas guardada en: {geopackage_output_path}")

    print("\nGenerando mapa de los grandes incendios...")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # 1. Dibujar TODOS los focos de calor en gris para dar contexto
    # zorder=2 significa que estará por encima del mapa base pero debajo de los puntos rojos
    gdf_cordillera.plot(ax=ax, marker='.', color='gray', markersize=5, alpha=0.5, label='Todos los focos de calor', zorder=2)
    
    # 2. Resaltar los puntos que pertenecen a los grandes incendios
    # zorder=3 para que estos puntos rojos se dibujen encima de los grises
    points_in_large_fires = gdf_cordillera[gdf_cordillera['fire_id'].isin(large_fires_gdf['fire_id'])]
    points_in_large_fires.plot(ax=ax, marker='o', color='red', markersize=15, label='Focos de Gran Incendio', zorder=3)
    
    # 3. Dibujar el límite del departamento al final para que quede por encima de todo
    # zorder=4 para asegurar que el borde negro esté por encima de todos los puntos
    departamento_gdf.to_crs(METRIC_CRS).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Dpto. Cordillera', zorder=4)

    # 4. Añadir mapa base (se dibuja por defecto en el fondo, zorder=1)
    cx.add_basemap(ax, crs=METRIC_CRS, source=cx.providers.CartoDB.Positron)
    ax.set_title(f"Grandes Incendios Forestales Identificados en Cordillera\n(Huella > {MIN_AREA_HA_FOR_LARGE_FIRE} ha y > {MIN_DETECTIONS_FOR_LARGE_FIRE} detecciones)", fontsize=16)
    ax.set_xlabel("Longitud (UTM)")
    ax.set_ylabel("Latitud (UTM)")
    
    # Recrear la leyenda en el orden correcto
    handles, labels = ax.get_legend_handles_labels()
    # Filtramos para quitar la etiqueta del borde del departamento si no la quieres
    order = [labels.index('Todos los focos de calor'), labels.index('Focos de Gran Incendio')]
    ax.legend([handles[i] for i in order], [labels[i] for i in order])

    # Guardar el mapa como una imagen
    map_output_path = OUTPUT_DIR / "mapa_grandes_incendios.png"
    plt.savefig(map_output_path, dpi=300, bbox_inches='tight')
    print(f"Mapa guardado en: {map_output_path}")

    # Mostrar el mapa en pantalla al final
    plt.show()

"""
Se han recopilado los polígonos de 14078 incendios forestales ocurridos en el Departamento de Cordillera entre 2018 y 2023, junto con la fecha de inicio de cada uno de ellos. Se identificaron 5053 eventos de incendio potenciales con las caracteristicas de Huella>100 ha y un grupo mayor a 15 detecciones 

1.  Filtrado Geográfico: Primero, carga todos los datos y se queda únicamente con los focos de calor que cayeron dentro de los límites del departamento de Cordillera.

2.  Agrupamiento Inteligente (Clustering): Luego, aplica su lógica principal: agrupa los puntos que están cercanos en el espacio (ej. a menos de 2 km) y en el tiempo (ej. en un lapso de 2 días) para tratarlos como un único "evento de incendio".

3.  Análisis de Eventos: Para cada "evento" agrupado, calcula sus características: su duración en días, el número total de detecciones y una estimación de su área máxima ("huella") en hectáreas.
"""