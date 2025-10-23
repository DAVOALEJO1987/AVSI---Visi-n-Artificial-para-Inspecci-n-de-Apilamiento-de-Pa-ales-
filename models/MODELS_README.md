# 🧠 Model Card – AVSI Best Model (`cnn_best.pt`)

**Proyecto:** Artificial Vision Stacking Inspection (AVSI)  
**Versión del modelo:** `cnn_best.pt`  
**Fecha de entrenamiento:** 2025-10-20  
**Autor:** David Alejandro Narváez Mejía / Francisco Javier Estupiñán Andrade  
**Profesor:** PhD. Gladys Villegas – Maestría en Inteligencia Artificial (UEES)  

---

## 📋 Descripción General

El modelo `cnn_best.pt` corresponde a la versión **óptima y final** del sistema de visión artificial desarrollado para el proyecto **AVSI**, cuyo propósito es la **verificación automática del apilamiento de pañales** previo al empaque industrial.  

Fue entrenado sobre un dataset propio de **1 000 imágenes** etiquetadas en dos clases:  
- `good_stack` (apilamiento correcto)  
- `bad_stack` (apilamiento defectuoso)

El modelo está basado en la arquitectura **ResNet-18**, utilizando **transfer learning** con pesos preentrenados en *ImageNet*.

---

## ⚙️ Configuración Técnica

| Parámetro | Valor |
|------------|-------|
| Arquitectura base | ResNet-18 |
| Framework | PyTorch 2.3.1 |
| Entrenamiento | Transfer Learning (ImageNet) |
| Dropout | 0.3 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Imagen de entrada | 224×224 RGB |
| Épocas entrenadas | 10 |
| Early stopping | Activado (paciencia = 3) |
| Data Augmentation | Rotación ±15°, flips, jitter, normalización RGB |
| Hardware | GPU RTX (CUDA 11.8) |

---

## 📊 Dataset Utilizado

| Dataset | Imágenes | Clases | Proporción | Procedencia |
|----------|-----------|---------|-------------|--------------|
| **Dataset principal (EVA ENGINEERING)** | 1 000 | 2 (`good_stack`, `bad_stack`) | 70/15/15 | EVA Engineering S.A. – IN34-EVA1200 REV 01 |

**Licencia del dataset:** uso académico y de investigación (no redistribuible).  
**Condiciones:** imágenes capturadas bajo entorno controlado de línea industrial.

---

## 🧪 Resultados del Entrenamiento

| Métrica | Valor | Descripción |
|----------|--------|-------------|
| **Accuracy** | 0.944 | Precisión global del modelo |
| **F1-score** | 0.945 | Balance entre precisión y recall |
| **mAP** | 0.984 | Precisión media promedio entre clases |
| **mIoU** | 0.870 | Superposición media entre clases |
| **FPS** | 144 | Velocidad de inferencia en tiempo real |

**Comparación con baseline (100 imágenes):**
| Modelo | Accuracy | mAP | mIoU | FPS |
|---------|-----------|------|------|------|
| `model_v1.pkl` | 0.89 | 0.91 | 0.76 | 110 |
| `cnn_best.pt` | **0.944** | **0.984** | **0.870** | **144** |

> ✅ El modelo final supera en más del **5%** la precisión del baseline y mantiene operación **en tiempo real (>30 FPS)**.

---

## 🧩 Arquitectura del Modelo

<img width="425" height="194" alt="image" src="https://github.com/user-attachments/assets/b29cd96a-7bbd-4182-896f-7778aa5a8b92" />


El modelo se guarda en formato **PyTorch (.pt)** con los pesos optimizados y las etiquetas de clase incluidas en el diccionario interno del checkpoint:

```python
torch.save({
    'model_state': model.state_dict(),
    'classes': ['good_stack', 'bad_stack']
}, 'models/cnn_best.pt')

## 🧩 LINK 50 MGAS
https://drive.google.com/file/d/1nDsgn6HSShq9KLmjwJSo__HI0x3v0XV6/view?usp=sharing

