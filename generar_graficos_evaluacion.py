import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Importaciones de Scikit-learn para los gráficos
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import CalibrationDisplay

MODEL_RESULTS_PKL = "resultados_modelo_para_eval.pkl"

# --- Carpeta de Salida ---
OUTPUT_DIR = Path("outputs_evaluacion_modelo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Los gráficos de evaluación del modelo se guardarán en: {OUTPUT_DIR}")


print(f"Cargando resultados del modelo desde '{MODEL_RESULTS_PKL}'...")
try:
    with open(MODEL_RESULTS_PKL, 'rb') as f:
        results = pickle.load(f)
    
    model = results['model']
    X_test = results['X_test']
    y_test = results['y_test']
    y_pred_proba = results['y_pred_proba']
    print("Resultados cargados exitosamente.")

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo '{MODEL_RESULTS_PKL}'.")
    print("Por favor, ejecuta tu script de modelado principal para generar este archivo.")
    exit()


# ==============================================================================
# GRÁFICO 4: CURVA DE PRECISIÓN-RECALL
# Objetivo: Evaluar el trade-off entre la precisión y la capacidad de detectar
#           todos los incendios reales. Esencial para problemas desbalanceados o
#           donde los positivos son muy importantes.
# ==============================================================================
print("Generando Gráfico 4: Curva de Precisión-Recall...")

# Calcular los puntos de la curva
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

# Calcular el Average Precision Score (un único número que resume la curva)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(recall, precision, linewidth=2, label=f'Modelo (AP = {avg_precision:.2f})')

# Estilo y etiquetas
ax.set_title('Curva de Precisión-Recall', fontsize=16)
ax.set_xlabel('Recall (Sensibilidad)', fontsize=12)
ax.set_ylabel('Precisión', fontsize=12)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.grid(True, linestyle='--')
ax.legend(loc='lower left')

# Guardar el gráfico
ruta_guardado_pr = OUTPUT_DIR / "grafico_eval_precision_recall.png"
plt.savefig(ruta_guardado_pr, dpi=300)
print(f"Gráfico guardado en: {ruta_guardado_pr}")
plt.show()


# ==============================================================================
# GRÁFICO 5: GRÁFICO DE CALIBRACIÓN
# Objetivo: Verificar si las probabilidades predichas por el modelo son fiables.
#           (Ej: Si el modelo dice "80% de riesgo", ¿realmente ocurre un incendio
#           el 80% de las veces en esas condiciones?)
# ==============================================================================
print("Generando Gráfico 5: Gráfico de Calibración...")

# Crear el gráfico
# Usamos la nueva API de Scikit-learn que lo hace todo automáticamente
fig, ax = plt.subplots(figsize=(8, 8))
display = CalibrationDisplay.from_predictions(
    y_test, 
    y_pred_proba, 
    n_bins=10, # Dividir las predicciones en 10 grupos de riesgo
    ax=ax
)

# Estilo y etiquetas
ax.set_title('Gráfico de Calibración del Modelo', fontsize=16)
ax.set_xlabel('Probabilidad Predicha (Riesgo Estimado)', fontsize=12)
ax.set_ylabel('Fracción de Positivos Real (Riesgo Observado)', fontsize=12)
ax.grid(True, linestyle='--')

# Guardar el gráfico
ruta_guardado_cal = OUTPUT_DIR / "grafico_eval_calibracion.png"
plt.savefig(ruta_guardado_cal, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {ruta_guardado_cal}")
plt.show()

print("\n¡Generación de gráficos de evaluación completada!")