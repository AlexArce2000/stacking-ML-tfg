## AHP

### Papers:

https://drive.google.com/drive/folders/1qJuH9IloP6aToYOSiW9RBPVVVJ_J4Ksl?usp=drive_link

El Proceso Analítico Jerárquico (Saaty, 1980) requiere:

1️⃣ Matriz de comparación por pares (n × n) \
2️⃣ Escala fundamental de Saaty (1–9) \
3️⃣ Cálculo del vector propio principal (pesos) \
4️⃣ Cálculo del Índice de Consistencia (CI) \
5️⃣ Cálculo del Ratio de Consistencia (CR) \  
6️⃣ Verificación: CR < 0.1 \  

### 1. Justificación de la prioridad del Recall (Sensibilidad)
En tu matriz, el Recall tiene una importancia de **5 (Fuerte)** sobre Accuracy y **7 (Muy Fuerte)** sobre Tiempo/Memoria.

*   **La Fuente:** *Jain, P., et al. (2020). "A review of machine learning applications in wildfire science and management".* ✔
*   **El Argumento:** Esta revisión exhaustiva establece que en la ciencia de incendios, la **omisión** (no detectar un fuego) tiene consecuencias irreversibles. La escala de Saaty de **5 a 7** se justifica porque el costo de un "Falso Negativo" incluye pérdida de biodiversidad y riesgo de vidas humanas, mientras que el costo de un "Falso Positivo" es meramente un costo logístico de desplazamiento.
*   **Cita para la tesis:** *"Siguiendo a Jain et al. (2020), la prioridad absoluta es la detección; por ello, en la matriz de Saaty se asigna una importancia fuerte (5) al Recall frente a la Exactitud, priorizando la seguridad sobre el desempeño estadístico general."*

### 2. Justificación del F1-Score y AUC sobre Accuracy
Asignaste un valor de **4 (Moderado a Fuerte)** al F1-Score frente al Accuracy.

*   **La Fuente:** *He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data".* ✔
*   **El Argumento:** Los autores demuestran que en conjuntos de datos con clases minoritarias (como los incendios forestales), el Accuracy es una métrica engañosa (Paradoja de la Exactitud). 
*   **El porqué del peso:** Se justifica un valor de **4** en Saaty porque el F1-Score es el balance crítico necesario para que el modelo sea útil. Un Accuracy del 99% podría no detectar ningún incendio, mientras que el F1-Score obliga al modelo a ser evaluado por su éxito real en la clase crítica.

### 3. Justificación de la baja importancia de Tiempo y Memoria
Asignaste valores de **1/5 a 1/7** (Importancia muy inferior).

*   **La Fuente:** *Giglio, L., et al. (2013). "Analysis of daily, monthly, and annual burned area using the fourth-generation Global Fire Emissions Database (GFED4)".* ✔
*   **El Argumento:** El monitoreo de incendios forestales se basa en ciclos satelitales (como MODIS o VIIRS) que entregan datos cada pocas horas.
*   **El porqué del peso:** Una diferencia de milisegundos en la inferencia de un modelo (como la que hay entre CatBoost y Stacking) es **irrelevante** frente al ciclo de actualización de los datos satelitales. Por eso, en la matriz de Saaty, estos criterios técnicos se califican como "muy inferiores" (1/7), ya que no afectan la respuesta operativa real.

### 4. Justificación de Precision y Specificity (Balance Logístico)
Asignaste una importancia de **1 (Igual)** o **2 (Débil)** entre ellos.

*   **La Fuente:** *Chuvieco, E., et al. (2010). "Development of a framework for fire risk assessment using remote sensing and GIS technology". Publicado en Remote Sensing of Environment.*" ✔
*   **El Argumento:** Chuvieco define los componentes del riesgo: Peligro (clima/combustible) y Vulnerabilidad. Argumenta que para que un modelo de riesgo sea útil, debe ser validado con métricas que aseguren que los recursos se muevan a donde realmente hay fuego. 
*   **El porqué del peso:** Si la Precision es baja, se pierden recursos en falsas alarmas. Si la Specificity es baja, no se pueden planificar quemas controladas o actividades preventivas en zonas seguras. Al ser ambos vitales para la **logística**, se justifica su peso equivalente o cercano en la escala de Saaty.

---

### Resumen de Referencias Bibliográficas para tu bibliografía:

1.  **Metodología AHP:** Saaty, T. L. (2008). *Relative Measurement and its Generalization in Decision Making*. Journal of the Real Academia de Ciencias. (Para validar la escala 1-9).✔  
2.  **Metodología TOPSIS:** Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag. (Para validar el ranking final).
3.  **Contexto Incendios:** Jain, P., et al. (2020). *A review of machine learning applications in wildfire science and management*. Environmental Reviews. (Para validar por qué el Recall manda).
4.  **Métricas ML:** He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data*. IEEE Transactions on Knowledge and Data Engineering. (Para validar por qué despreciaste el Accuracy).

**Consejo para el examen oral:** Si el jurado te pregunta: *"¿Por qué puso un 7 y no un 9?"*, tu respuesta debe ser: *"Se eligió 7 (Muy Fuerte) porque, aunque la detección es crítica, un valor de 9 (Extrema importancia) anularía por completo el peso de los demás criterios, y para una gestión eficiente también necesitamos considerar la logística (Precision) y la viabilidad técnica"*.