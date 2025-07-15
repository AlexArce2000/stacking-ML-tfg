import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx 

def dibujar_incendios_por_fecha(csv_path, fecha_especifica, shp_area_estudio_path=None):
    """
    Dibuja los focos de incendio de una fecha específica sobre un mapa de fondo.

    Args:
        csv_path (str): Ruta al archivo CSV que contiene los focos de incendio.
        fecha_especifica (str): La fecha que se desea visualizar, en formato 'YYYY-MM-DD'.
        shp_area_estudio_path (str, optional): Ruta a un shapefile para dibujar un contorno de referencia.
    """
    print(f"Cargando datos desde: {csv_path}")
    try:
        df_focos = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"¡ERROR! No se encontró el archivo en la ruta: {csv_path}")
        return

    df_focos['acq_date'] = pd.to_datetime(df_focos['acq_date'])

    print(f"Filtrando los incendios ocurridos en la fecha: {fecha_especifica}")
    fecha_obj = pd.to_datetime(fecha_especifica).date()
    df_filtrado = df_focos[df_focos['acq_date'].dt.date == fecha_obj]

    if df_filtrado.empty:
        print(f"RESULTADO: No se encontraron focos de incendio para la fecha {fecha_especifica}.")
        return

    print(f"RESULTADO: Se encontraron {len(df_filtrado)} focos de incendio.")

    gdf_incendios = gpd.GeoDataFrame(
        df_filtrado,
        geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude),
        crs="EPSG:4326"  
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    gdf_incendios.to_crs(epsg=3857).plot(
        ax=ax,
        marker='o',         # Forma del punto
        color='red',        # Color del punto
        markersize=50,      # Tamaño del punto
        alpha=0.7,          # Transparencia
        edgecolor='black',  # Color del borde del punto
        label=f'Incendios ({fecha_especifica})'
    )
    
    if shp_area_estudio_path:
        try:
            area_estudio = gpd.read_file(shp_area_estudio_path)
            area_estudio.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='Límite Dpto.')
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el shapefile del área de estudio: {e}")


    print("Añadiendo mapa base de calles y geografía...")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom=10)

    ax.set_title(f"Focos de Incendio Detectados el {fecha_especifica}", fontsize=16)
    ax.set_axis_off() 
    ax.legend()
    plt.tight_layout()
    plt.show()


ruta_csv_incendios = "data/FOCOS/focos_cordillera_viirs.csv"

fecha_a_dibujar = "2023-08-15"

ruta_shp_cordillera = "data/COORDILLERA/Departamento_Coordillera.shp"

dibujar_incendios_por_fecha(
    csv_path=ruta_csv_incendios, 
    fecha_especifica=fecha_a_dibujar,
    shp_area_estudio_path=ruta_shp_cordillera
)