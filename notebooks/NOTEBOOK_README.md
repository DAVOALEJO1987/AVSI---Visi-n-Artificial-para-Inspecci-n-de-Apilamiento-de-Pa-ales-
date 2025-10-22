# 🧠 Explicación Técnica del Proyecto AVSI
**Artificial Vision Stacking Inspection (AVSI)**  
*Sistema de Visión por Computador e Inteligencia Artificial para inspección de apilamiento de pañales.*

---

## 🔹 1. Contexto General
El proyecto **AVSI** fue desarrollado como parte de la Maestría en Inteligencia Artificial (UEES) y representa una solución integral de **visión por computador** para el **control automatizado del apilamiento de pañales** en líneas industriales.  
El objetivo es **garantizar la precisión y uniformidad del apilamiento**, reduciendo errores humanos, riesgos ergonómicos y costos operativos.

La solución fue implementada utilizando **Python**, **PyTorch** y **Streamlit**, integrando los componentes clásicos de un pipeline de IA:
1. Análisis exploratorio de datos (EDA)  
2. Preprocesamiento de imágenes  
3. Modelado mediante CNN con *transfer learning*  
4. Optimización de hiperparámetros  
5. Evaluación de métricas  
6. Despliegue en una interfaz interactiva.

---

## 🔹 2. Estructura del Proyecto
El proyecto se desarrolló en forma modular bajo una estructura estandarizada compatible con GitHub:


Cada notebook cumple una función dentro del **ciclo de vida del modelo** y se conecta con los directorios:
- `/data/raw` → Imágenes originales (100 y 1 000).  
- `/data/processed` → Datos limpios y balanceados.  
- `/models` → Pesos del modelo (`best_model.pt`).  
- `/results` → Métricas, figuras y reportes.  
- `/app` → Interfaz desarrollada en **Streamlit**.

---

## 🔹 3. Flujo Técnico Detallado

### 🧩 **01_exploracion_AVSI.ipynb**
- Realiza el **análisis exploratorio (EDA)** sobre los datasets de 100 y 1 000 imágenes.  
- Extrae **resoluciones**, **niveles de brillo y contraste**, **duplicados** mediante *perceptual hashing* y distribuciones de clase.  
- Permite visualizar la calidad del dataset y definir estrategias de limpieza.

### ⚙️ **02_preprocesamiento_AVSI.ipynb**
- Implementa la limpieza y normalización de imágenes (224×224 px, RGB).  
- Aplica **técnicas de data augmentation**: rotaciones, flips y jitter.  
- Divide los datos en **train/val/test** de manera estratificada.  
- Exporta la estructura a `/data/processed/` y un archivo `labels.csv` con metadatos.

### 🧠 **03_modelado_AVSI.ipynb**
- Entrena una **ResNet-18** preentrenada (*transfer learning*).  
- Congela las capas del backbone, reemplaza la capa final y optimiza con **Adam**.  
- Implementa *early stopping* y guarda el mejor modelo (`best_model.pt`).  
- Genera curvas de **pérdida** y **exactitud**.

### 🔬 **04_optimizacion_AVSI.ipynb**
- Realiza una **búsqueda en malla (Grid Search)** variando:
  - *Learning rate*
  - *Weight decay*
  - *Batch size*
  - *Freeze_backbone*
- Calcula sensibilidad y promedio de validación (`val_acc`) para cada hiperparámetro.  
- Almacena la mejor configuración y métricas comparativas.

### 📈 **05_evaluacion_AVSI.ipynb**
- Consolida resultados del entrenamiento y optimización.  
- Calcula métricas finales: **accuracy, matriz de confusión, clasificación por clase.**  
- Genera un **reporte ejecutivo (final_report.txt)** con KPIs clave.  
- Resume mejoras de rendimiento tras la ampliación del dataset.

---

Cada notebook cumple una función dentro del **ciclo de vida del modelo** y se conecta con los directorios:
- `/data/raw` → Imágenes originales (100 y 1 000).  
- `/data/processed` → Datos limpios y balanceados.  
- `/models` → Pesos del modelo (`best_model.pt`).  
- `/results` → Métricas, figuras y reportes.  
- `/app` → Interfaz desarrollada en **Streamlit**.

---

## 🔹 4. Componentes Técnicos Principales

| Módulo | Descripción |
|--------|--------------|
| **Framework de IA** | PyTorch 2.3.1 (entrenamiento y evaluación del modelo CNN) |
| **Arquitectura Base** | ResNet-18 (transfer learning con pesos preentrenados en ImageNet) |
| **Interfaz** | Streamlit (visualización e interacción en tiempo real) |
| **Lenguaje** | Python 3.10 |
| **Dependencias** | NumPy, OpenCV, Pillow, scikit-learn, Matplotlib, TorchVision |
| **Almacenamiento** | Directorios `/data/`, `/models/`, `/results/` y `labels.csv` |
| **Hardware** | Compatible con CPU o GPU (CUDA 11+) |

---

## 🔹 5. Indicadores de Desempeño (KPIs)

| Métrica | Resultado | Observaciones |
|----------|------------|----------------|
| **mAP** | ≥ 0.70 | Mejor desempeño con dataset ampliado (1 000 imágenes). |
| **IoU** | > 0.50 | Alineación adecuada en apilamientos detectados. |
| **FPS** | > 30 | Procesamiento en tiempo real (industrial-ready). |
| **Reducción manual** | 70 % | Disminución significativa en intervención humana. |

---

## 🔹 6. Conclusión Técnica
El sistema **AVSI** integra un flujo completo de IA, desde la adquisición de datos hasta el despliegue interactivo.  
Su diseño modular permite **replicar, escalar y adaptar** el modelo a otros productos industriales (jarras, cajas, botellas, etc.).  
La integración con **Streamlit** habilita un entorno accesible para usuarios no técnicos y garantiza una implementación robusta bajo los principios de la **Industria 4.0**.

---

## 📎 Archivos Relacionados
- `requirements.txt` — dependencias del proyecto.  
- `.gitignore` — exclusión de datos y modelos grandes.  
- `/docs/consideraciones_eticas.md` — análisis ético completo.  
- `/results/final_report.txt` — resumen de KPIs.  
- `/models/best_model.pt` — modelo entrenado final.

---

**Autor:** David Alejandro Narváez Mejia / Francisco Javier Estupiñan Andrade  
**Institución:** Universidad de Especialidades Espíritu Santo (UEES)  
**Empresa:** EVA Engineering S.A.  
**Licencia:** MIT  
