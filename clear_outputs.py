import os
import shutil

def vaciar_carpeta_outputs(carpeta='outputs'):
    if os.path.exists(carpeta):
        for elemento in os.listdir(carpeta):
            ruta_elemento = os.path.join(carpeta, elemento)
            try:
                if os.path.isfile(ruta_elemento) or os.path.islink(ruta_elemento):
                    os.unlink(ruta_elemento)  # elimina archivos o enlaces simbólicos
                elif os.path.isdir(ruta_elemento):
                    shutil.rmtree(ruta_elemento)  # elimina carpetas recursivamente
            except Exception as e:
                print(f"No se pudo eliminar {ruta_elemento}. Error: {e}")
        print(f'Carpeta "{carpeta}" vaciada correctamente.')
    else:
        print(f'La carpeta "{carpeta}" no existe.')

# Llamar a la función
vaciar_carpeta_outputs()
