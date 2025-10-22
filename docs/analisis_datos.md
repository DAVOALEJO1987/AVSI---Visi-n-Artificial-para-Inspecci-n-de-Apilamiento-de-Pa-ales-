# 🔍 ANÁLISIS DE DATOS – PROYECTO AVSI
**Artificial Vision Stacking Inspection (AVSI)**  
*Visión por Computador aplicada a la inspección automática del apilamiento de pañales.*

---

## 🧩 1. Descripción detallada del Dataset

El dataset empleado para el desarrollo del sistema AVSI fue construido con imágenes reales capturadas en líneas de empaque industrial, licenciadas por **EVA ENGINEERING S.A.** bajo el código **IN34-EVA1200 REV 01**.  
Está compuesto por **dos conjuntos de datos principales**:

| Dataset | Nº Imágenes | Descripción | Propósito |
|----------|--------------|--------------|------------|
| **Dataset 1 (RAW100)** | 100 | Imágenes originales capturadas en planta, con variaciones de iluminación y posición. | Baseline inicial para la arquitectura. |
| **Dataset 2 (RAW1000)** | 1 000 | Conjunto extendido mediante *data augmentation* (rotaciones, flips, jitter, variaciones de brillo y contraste). | Entrenamiento final optimizado. |

Cada imagen está etiquetada en dos clases:
- `good_stack` → apilamiento correcto.  
- `bad_stack` → apilamiento incorrecto o defectuoso.

Las imágenes presentan **resolución uniforme de 640×640 px**, formato **JPG**, profundidad de color **RGB (8 bits)** y tamaño promedio de **180 KB**.  
El conjunto fue reorganizado en `/data/raw/` y posteriormente procesado a `/data/processed/` con divisiones **train 70 %**, **val 15 %**, **test 15 %**.

---

## 📊 2. Estadísticas descriptivas del dataset

| Variable | Mínimo | Máximo | Promedio | Desviación | Observaciones |
|-----------|--------|---------|-----------|-------------|----------------|
| Ancho (px) | 640 | 640 | 640 | 0 | Resolución constante |
| Alto (px) | 640 | 640 | 640 | 0 | Proporción cuadrada |
| Brillo (mean_gray) | 72 | 189 | 128.5 | 22.1 | Variación de iluminación |
| Contraste (std_gray) | 35 | 79 | 56.8 | 9.4 | Suficiente para segmentación |
| Tamaño archivo (KB) | 120 | 260 | 178 | 38 | Correlacionado con brillo |
| Desenfoque (blur_var) | 50 | 2 400 | 890 | 530 | Algunas imágenes borrosas |
| Clases | good_stack / bad_stack | — | — | — | Dataset binario |

La exploración inicial confirmó que el conjunto es **pequeño y desbalanceado**, por lo cual se aplicaron técnicas de aumento y balanceo sintético.

<img width="685" height="751" alt="image" src="https://github.com/user-attachments/assets/8c3658f2-5643-42ae-b9fd-ad6d884e77c7" />

<img width="669" height="746" alt="image" src="https://github.com/user-attachments/assets/782c2c7e-f4e3-4167-a994-b80e8cf593ca" />

---

## 📈 3. Visualizaciones del EDA

Durante la fase de EDA, implementada en `01_exploracion_AVSI.ipynb`, se generaron las siguientes visualizaciones:

- **Histogramas de resolución** → validación de tamaño uniforme.  
- **Histogramas de brillo y contraste** → detección de imágenes subexpuestas.  
- **Distribución por clase (bar plot)** → evidencia de desbalance a favor de *good_stack*.  
- **Matriz de correlación RGB** → redundancia cromática entre canales R, G y B.  
- **Mapas PCA/t-SNE (embeddings)** → separación parcial entre clases basada en textura y patrón visual.  
- **Gráfico de dispersión tamaño-brillo** → correlación positiva; imágenes más claras ocupan mayor tamaño.  

Estas visualizaciones orientaron la definición del pipeline de limpieza y normalización.

---

## 🔎 4. Identificación de patrones y correlaciones

- **Correlaciones fuertes**:  
  `mean_r`, `mean_g`, `mean_b` presentan **r > 0.9**, lo que evidencia **colinealidad cromática**.  
  Se recomienda conservar una sola variable promedio (`mean_gray`).

- **Patrones visuales detectados**:  
  - Imágenes *bad_stack* tienden a mostrar **mayor desenfoque** (blur > 1500).  
  - *good_stack* conserva **bordes definidos y contraste más alto**.  
  - Brillos extremos se asocian a fondos claros que inducen falsos positivos.  

- **Embeddings ResNet-18 (PCA)**: las dos clases muestran agrupamientos diferenciados, confirmando la **viabilidad de clasificación supervisada**.

---

## ⚠️ 5. Outliers y anomalías detectadas

| Tipo de anomalía | Descripción | Acción tomada |
|------------------|-------------|----------------|
| **Imágenes duplicadas** | Detección vía *perceptual hash* (pHash). | Eliminadas automáticamente. |
| **Baja exposición** | Brillo < 80, contraste < 30. | Filtradas o corregidas con normalización de histograma. |
| **Desenfoque extremo** | blur_var > 2000. | Descartadas del set de entrenamiento. |
| **Desbalance de clases** | 65 % good_stack vs 35 % bad_stack. | *Data augmentation* focalizado en clases minoritarias. |

---

## 🧠 6. Decisiones de Preprocesamiento Justificadas

1. **Redimensionamiento uniforme (224×224 px):**  
   Permite compatibilidad con ResNet-18 y reduce la varianza geométrica.
2. **Normalización RGB (canales 0–1, luego z-score con medias de ImageNet):**  
   Asegura convergencia estable durante el entrenamiento.
3. **Data Augmentation controlado:**  
   - Rotaciones aleatorias ±15°.  
   - Flips horizontales.  
   - *Color jitter* en brillo y contraste ±10 %.  
   Mejora la generalización y compensa la falta de datos reales.
4. **Balanceo mediante SMOTE (opcional):**  
   En caso de dataset tabular derivado de features (hog, hist, embeddings).  
5. **Eliminación de duplicados/outliers:**  
   Basado en hash perceptual y umbral de desenfoque.  
6. **División estratificada train/val/test:**  
   Garantiza representatividad de ambas clases en cada subconjunto.

---

## 📉 7. Manejo de Datos Faltantes o Desbalanceados

- **Datos faltantes:** no se registraron archivos corruptos o rutas inválidas tras la limpieza inicial.  
  En caso de errores de lectura (`cv2.imread(None)`), se descartan automáticamente.  
- **Datos desbalanceados:**  
  - Clase `bad_stack` incrementada mediante *augmentation*.  
  - Ensayos alternativos con **oversampling SMOTE** en embeddings de características.  
  - Métricas ajustadas con **F1-score** y **balanced accuracy** para evitar sesgo de precisión.

---

## 📘 8. Conclusiones del EDA

1. El dataset cumple condiciones suficientes para entrenamiento de modelos CNN simples.  
2. Existen correlaciones cromáticas redundantes que se mitigan mediante normalización.  
3. El desbalance de clases fue corregido exitosamente con *data augmentation*.  
4. El desenfoque y la variabilidad de iluminación constituyen los factores críticos a controlar.  
5. Las decisiones de limpieza y estandarización incrementaron la estabilidad del modelo y la reproducibilidad experimental.

---

## 📎 9. Referencias

1. N. Hütten et al., “Deep Learning for Automated Visual Inspection in Industrial Applications,” *Machines*, 2024.  
2. A. Wan et al., “Deep learning-based intelligent visual inspection for defect detection,” *Engineering Applications of AI*, 2025.  
3. J. Huang, “Automated Logistics Packaging Inspection Based on Deep Learning,” *Traitement du Signal*, 2025.  
4. J. Villalba-Diez et al., “Deep Learning for Industrial Quality Control,” *Sensors*, 2019.  
5. PhD. Gladys Villegas, UEES – Materia MIAR0545 – Proyecto Integrador 2025.

---
