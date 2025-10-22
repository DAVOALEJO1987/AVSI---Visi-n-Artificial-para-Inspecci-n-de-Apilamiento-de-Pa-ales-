# 🤖 AVSI — Artificial Vision Stacking Inspection
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Build-Stable-success.svg)]()

## 📄 Descripción técnica del proyecto
**AVSI (Artificial Vision Stacking Inspection)** es un sistema de **visión por computador e inteligencia artificial** desarrollado para inspeccionar automáticamente el apilamiento de pañales en líneas de producción industrial.  
Mediante una **red neuronal convolucional (ResNet-18)** y técnicas de **transfer learning**, el sistema identifica apilamientos correctos e incorrectos en tiempo real, con una precisión superior al 94% y una velocidad mayor a 30 FPS.  
El proyecto integra un pipeline completo de datos, entrenamiento y evaluación con una interfaz gráfica en **Streamlit**, orientado a la mejora de calidad y reducción de errores humanos en procesos de manufactura.

**Profesor:** PhD. Gladys Villegas  
**Autores:** Francisco Javier Estupiñán Andrade / David Alejandro Narváez Mejía  

---

## 📚 Tabla de Contenidos
1. [Descripción del Problema](#-descripción-del-problema)  
2. [Dataset](#-dataset)  
3. [Metodología](#-metodología)  
4. [Resultados](#-resultados)  
5. [Instalación y Uso](#-instalación-y-uso)  
6. [Interfaz de Usuario](#-interfaz-de-usuario)  
7. [Estructura del Proyecto](#-estructura-del-proyecto)  
8. [Consideraciones Éticas](#-consideraciones-éticas)  
9. [Autores y Contribuciones](#-autores-y-contribuciones)  
10. [Licencia](#-licencia)  
11. [Agradecimientos y Referencias](#-agradecimientos-y-referencias)  

---

## 🧩 Descripción del Problema

### ¿Qué problema resuelve el proyecto?
El sistema AVSI automatiza el proceso de **inspección del apilamiento de pañales**, que normalmente se realiza de forma manual. Este proceso es propenso a errores humanos, inconsistencias visuales y riesgos ergonómicos.  

### ¿Por qué es importante?
En entornos industriales de alta velocidad, incluso pequeñas fallas de apilamiento pueden generar pérdidas económicas significativas, rechazos de producto y fallas en el empaque final. AVSI permite detectar estos defectos en tiempo real con alta precisión, reduciendo costos y mejorando la trazabilidad de calidad.  

### ¿Quiénes son los usuarios objetivo?
Ingenieros de control de calidad, técnicos de automatización y operadores de líneas industriales dentro del sector de productos higiénicos, alimentos y empaques automáticos.

---

## 📊 Dataset

### Descripción de los datos utilizados
El sistema emplea **dos datasets propios** desarrollados por **EVA ENGINEERING S.A.**:
| Dataset | Nº de Imágenes | Descripción | Propósito |
|----------|----------------|--------------|------------|
| Dataset 1 | 100 | Imágenes originales capturadas en planta bajo distintas condiciones. | Baseline inicial. |
| Dataset 2 | 1 000 | Dataset ampliado con *data augmentation*. | Entrenamiento final optimizado. |

<img width="839" height="614" alt="image" src="https://github.com/user-attachments/assets/d93db7fb-8a93-47b0-84dc-e30338161084" />


### Fuente y licencia de los datos
- **Procedencia:** EVA ENGINEERING S.A. (código IN34-EVA1200 REV 01).  
- **Licencia:** Uso académico y prototipado interno.  
- **Condición:** Prohibida su distribución sin autorización.  

### Características principales
- Formato: JPG, resolución 640×640 px  
- Clases: `good_stack` / `bad_stack`  
- Balanceo: Estratificado 70/15/15  
- Calidad: Imágenes industriales RGB 8-bit  

---

## 🧠 Metodología

### Tipo de modelo utilizado y justificación
Se seleccionó una **ResNet-18** con *transfer learning* desde **ImageNet**, por su equilibrio entre precisión, velocidad y eficiencia en entornos industriales.  
Se reemplazó la capa final (`fc`) por una densa binaria para clasificación y se aplicó *dropout (0.3)* para regularización.

### Preprocesamiento aplicado
- Redimensionamiento a 224×224 px  
- Normalización RGB con medias de ImageNet  
- Data augmentation: rotaciones ±15°, flips, jitter de brillo y contraste  
- Balanceo mediante **WeightedRandomSampler**

### Técnicas de optimización empleadas
- Grid Search sobre hiperparámetros (`lr`, `batch_size`, `dropout`, `optimizer`)  
- Regularización por *early stopping* y *weight decay*  
- Entrenamiento con **Adam (lr=0.0005)** y validación cruzada  

### Métricas de evaluación seleccionadas
- **Accuracy**, **F1-Score**, **mAP (mean Average Precision)**  
- **mIoU (Intersection over Union)**  
- **FPS (velocidad de inferencia)**

---

## 📈 Resultados

| Dataset | Accuracy | F1-Score | mAP | mIoU | FPS |
|----------|-----------|-----------|------|------|------|
| Dataset 1 (100 imágenes) | 0.89 | 0.90 | 0.91 | 0.76 | 110 |
| Dataset 2 (1000 imágenes) | **0.944** | **0.945** | **0.984** | **0.870** | **144** |

**Comparación:**  
La expansión del dataset y la optimización de hiperparámetros incrementaron la precisión en **+1.2 %** y la estabilidad del modelo, manteniendo tiempo real (>30 FPS).

**Gráficos:**
**DATASET 1**
<img width="675" height="746" alt="image" src="https://github.com/user-attachments/assets/459e6f7c-1206-4c6f-a187-bbca281666a6" />

**DATASET 2**
<img width="659" height="752" alt="image" src="https://github.com/user-attachments/assets/2c8b1897-42e5-4c0f-8c1d-f1b6290ec4fc" />

---

## ⚙️ Instalación y Uso

### Requisitos del sistema
- **Python 3.10+**
- GPU compatible con CUDA 11+ (opcional)
- 8 GB de RAM mínimo
- SO: Windows 11 / Ubuntu 22.04

### Instalación
```bash
git clone https://github.com/usuario/AVSI.git
cd AVSI
pip install -r requirements.txt
# 1️⃣ Procesar datos
python -m src.data_processing

# 2️⃣ Entrenar el modelo
python -m src.train --epochs 10 --lr 0.0005

# 3️⃣ Evaluar resultados
python -m src.evaluate
[10/10] train_acc=0.94 | val_acc=0.945
Matriz de confusión y métricas guardadas en /results/metrics/
