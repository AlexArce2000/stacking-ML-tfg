import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import os
import glob # <--- 1. IMPORTANTE: Módulo para buscar archivos

def dibujar_incendios_desde_fuente_firms(firms_data_path, shp_area_estudio_path=None):
    """
    Carga todos los archivos CSV de datos de incendios desde la fuente (FIRMS),
    los combina y los dibuja en un mapa.

    Args:
        firms_data_path (str): Ruta a la carpeta principal que contiene los datos de FIRMS.
        shp_area_estudio_path (str, optional): Ruta al shapefile del área de estudio.
    """
    search_pattern = os.path.join(firms_data_path, '**', '*.csv')
    print(f"Buscando archivos CSV de FIRMS en: {search_pattern}")
    
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        print(f"¡ERROR! No se encontraron archivos CSV en la ruta: {firms_data_path}")
        return

    print(f"Archivos encontrados: {len(csv_files)}")
    
    list_of_dfs = []
    for file in csv_files:
        print(f"  -> Cargando {os.path.basename(file)}")
        df = pd.read_csv(file)
        list_of_dfs.append(df)

    df_focos_total = pd.concat(list_of_dfs, ignore_index=True)
    

    if df_focos_total.empty:
        print("RESULTADO: No se encontraron focos de incendio en los archivos.")
        return
    else:
        print(f"\nRESULTADO: Se encontraron {len(df_focos_total)} focos de incendio en total.")
        
        gdf_incendios = gpd.GeoDataFrame(
            df_focos_total,
            geometry=gpd.points_from_xy(df_focos_total.longitude, df_focos_total.latitude),
            crs="EPSG:4326"
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        if shp_area_estudio_path:
            try:
                area_estudio = gpd.read_file(shp_area_estudio_path)
                area_estudio.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='Límite Dpto. Cordillera')
            except Exception as e:
                print(f"Advertencia: No se pudo cargar el shapefile: {e}")

        gdf_incendios.to_crs(epsg=3857).plot(
            ax=ax, marker='o', color='red', markersize=15, alpha=0.5,
            edgecolor='black', linewidth=0.5, label='Incendios (Fuente FIRMS)'
        )
        
        ax.set_title("Historial de Focos de Incendio (Datos FIRMS)", fontsize=18)
        ax.legend()

    print("Añadiendo mapa base...")
    if 'ax' in locals():
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom=10)
        ax.set_axis_off()
        plt.tight_layout()

        output_dir = "outputs_focos_reales"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = "mapa_incendios_historico_desde_FIRMS.png"
        output_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"¡Mapa guardado exitosamente en: {output_path}!")
        
        plt.show()

# --- PARÁMETROS DE EJECUCIÓN ---
# Ahora la ruta principal apunta a la carpeta que contiene las descargas de FIRMS
ruta_datos_firms = "data/FIRMS" 
ruta_shp_cordillera = "data/COORDILLERA/Departamento_Coordillera.shp"

# --- LLAMADA A LA FUNCIÓN ---
dibujar_incendios_desde_fuente_firms(
    firms_data_path=ruta_datos_firms,
    shp_area_estudio_path=ruta_shp_cordillera
)