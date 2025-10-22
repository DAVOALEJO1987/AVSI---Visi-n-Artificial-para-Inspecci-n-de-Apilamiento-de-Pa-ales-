# 🏗️ ARQUITECTURA DEL SISTEMA AVSI
**Artificial Vision Stacking Inspection (AVSI)**  
*Sistema de visión artificial basado en redes neuronales convolucionales (CNN) para la verificación automática del apilamiento de pañales.*

---

## 🧩 1. Tipo de Modelo Seleccionado y Justificación

### **Modelo Principal: ResNet-18 (Transfer Learning)**
El sistema AVSI utiliza una **arquitectura ResNet-18** como red base para clasificación binaria (apilamiento correcto vs. incorrecto).  
ResNet (Residual Network) es una **CNN profunda** que incorpora *skip connections*, lo que permite un entrenamiento más estable y evita el problema de *vanishing gradients* en redes profundas.

**Justificación técnica:**
- **Transfer Learning:** Al usar pesos preentrenados en **ImageNet**, se aprovecha conocimiento previo de millones de imágenes, mejorando la generalización del modelo con datasets pequeños (100 y 1000 imágenes).
- **Rendimiento y eficiencia:** ResNet-18 ofrece un equilibrio óptimo entre **precisión (>90%)** y **velocidad (>30 FPS)**, lo cual es esencial para procesos industriales en tiempo real.
- **Compatibilidad industrial:** Su bajo costo computacional permite ejecución en dispositivos de borde como **Raspberry Pi** o **Jetson Nano**, facilitando la integración con celdas robóticas o sistemas PLC.
- **Escalabilidad:** La arquitectura permite extender fácilmente el modelo hacia ResNet-34, MobileNetV2 o YOLOv8 si el proyecto evoluciona hacia detección en lugar de clasificación.

---

## ⚙️ 2. Arquitectura Detallada del Sistema

El flujo general del sistema AVSI combina procesamiento de datos, entrenamiento del modelo y despliegue de inferencias.  
La arquitectura está organizada en **cinco capas funcionales:**

### **1️⃣ Capa de Adquisición**
- Captura imágenes mediante cámara industrial **SVPRO 4K** conectada a Raspberry Pi o laptop de desarrollo.
- Genera datasets almacenados en `/data/raw/` con etiquetas *good_stack* y *bad_stack*.
- Las imágenes se preprocesan en formato RGB con resolución estándar de **224×224 px**.

### **2️⃣ Capa de Preprocesamiento**
Implementada en el notebook `02_preprocesamiento_AVSI.ipynb`, incluye:
- **Redimensionamiento y normalización:** Se homogeniza el tamaño y la escala de color.  
- **Data Augmentation:** Rotaciones, flips y jitter de brillo/contraste para ampliar la muestra.  
- **División estratificada:** 70% entrenamiento, 15% validación, 15% prueba.  
- **Salida estructurada:** `/data/processed/train`, `/val`, `/test`, y `labels.csv`.

### **3️⃣ Capa de Modelado**
Desarrollada en `03_modelado_AVSI.ipynb`:
- Base: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`
- Se reemplaza la capa final (`fc`) con una densa de salida binaria (`Linear(in_features, 2)`).
- Optimizador **Adam**, función de pérdida **CrossEntropyLoss**, y *early stopping* para prevenir sobreajuste.
- Guardado del mejor modelo como `models/best_model.pt`.

### **4️⃣ Capa de Optimización y Validación**
En `04_optimizacion_AVSI.ipynb`:
- **Grid Search** sobre hiperparámetros: *lr*, *batch_size*, *weight_decay*, *freeze_backbone*.  
- Análisis de sensibilidad y comparación de rendimiento.  
- Selección automática del modelo con mayor *val_acc*.

### **5️⃣ Capa de Despliegue**
El modelo final se integra en una aplicación **Streamlit**, que permite:
- Cargar imágenes individuales o en lote.  
- Mostrar predicciones con probabilidades.  
- Exportar resultados y visualizar métricas de confianza.  
- Procesamiento en tiempo real (>30 FPS).

---

## 🔄 3. Pipeline de Datos (de Input a Output)

El flujo de información del sistema puede resumirse en el siguiente pipeline:

<img width="1163" height="614" alt="Accidentalidad - visual selection" src="https://github.com/user-attachments/assets/6f75a9b4-c221-417e-9883-18d59dc15e44" />

---

## 🧠 4. Tecnologías y Librerías Utilizadas

### **Lenguaje Base**
- **Python 3.10.14**

### **Librerías Principales**
| Categoría | Librería | Versión | Descripción |
|------------|-----------|----------|-------------|
| **IA y Deep Learning** | `torch` | 2.3.1 | Framework de redes neuronales |
|  | `torchvision` | 0.18.1 | Modelos preentrenados (ResNet-18) |
| **Procesamiento de imágenes** | `opencv-python` | 4.10.0.84 | Lectura, redimensionamiento y visualización |
|  | `Pillow` | 10.4.0 | Manipulación avanzada de imágenes |
| **Ciencia de Datos** | `numpy` | 1.26.4 | Cálculos numéricos y operaciones matriciales |
|  | `pandas` | 2.2.2 | Manipulación de estructuras tabulares |
|  | `scikit-learn` | 1.5.2 | Split de datasets y métricas de evaluación |
| **Visualización** | `matplotlib` | 3.9.2 | Gráficos de entrenamiento y métricas |
|  | `plotly` | 5.23.0 | Visualizaciones interactivas |
| **Interfaz** | `streamlit` | 1.37.1 | Despliegue de interfaz interactiva web |
| **Optimización y utilidades** | `tqdm` | 4.66.4 | Barra de progreso en entrenamiento |
|  | `joblib` | 1.4.2 | Guardado de objetos y modelos |
| **Configuración** | `pyyaml` | 6.0.2 | Gestión de archivos de configuración |

### **Hardware y Entorno**
- Laptop con CPU Intel i7 (8 núcleos) o Raspberry Pi 5 (8 GB RAM).  
- GPU opcional con soporte CUDA 11+ (NVIDIA).  
- Sistema operativo: Ubuntu 22.04 / Windows 11.  
- Entorno de desarrollo: **Visual Studio Code + Jupyter Notebook**.  

---

## 🔍 5. Resumen Arquitectónico

| Componente | Descripción Técnica |
|-------------|--------------------|
| **Dataset** | Imágenes RGB (224x224 px), 2 clases (good_stack / bad_stack) |
| **Modelo** | CNN ResNet-18 con *transfer learning* |
| **Entrenamiento** | CrossEntropyLoss, optimizador Adam, early stopping |
| **Evaluación** | Accuracy, Confusion Matrix, Classification Report |
| **Optimización** | Grid Search (lr, batch_size, weight_decay) |
| **Interfaz** | Streamlit para predicción e interpretación visual |
| **Desempeño esperado** | mAP ≥ 0.70, IoU > 0.50, FPS > 30 |

---

## 📈 6. Conclusión
La arquitectura técnica de **AVSI** fue diseñada bajo los principios de **modularidad, reproducibilidad y eficiencia industrial**, permitiendo su integración futura con sistemas de control y automatización (PLC, SCADA o celdas robotizadas).  
El uso de **ResNet-18** con *transfer learning* asegura un equilibrio entre desempeño y costo computacional, posicionando el sistema como un **prototipo sólido para la industria 4.0**.

---
