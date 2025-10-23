# 🤖 AVSI — Artificial Vision Stacking Inspection
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Build-Stable-success.svg)]()

## 📄 Descripción técnica del proyecto
**AVSI (Artificial Vision Stacking Inspection)** es un sistema de **visión por computador e inteligencia artificial** desarrollado para inspeccionar automáticamente el apilamiento de pañales en líneas de producción industrial.  
Mediante una **red neuronal convolucional (ResNet-18)** y técnicas de **transfer learning**, el sistema identifica apilamientos correctos e incorrectos en tiempo real, con una precisión superior al 94% y una velocidad mayor a 30 FPS.  
El proyecto integra un pipeline completo de datos, entrenamiento y evaluación con una interfaz gráfica en **Streamlit**, orientado a la mejora de calidad y reducción de errores humanos en procesos de manufactura.

##  Descripción grafica 
<img width="1536" height="1024" alt="ChatGPT Image Oct 16, 2025, 06_36_02 PM" src="https://github.com/user-attachments/assets/ae897a3e-3e11-4417-abfd-d7236dca71e2" />
**Fuente:** [AVSI / Generado por ChatGPT]


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

##  Arquitectura Vision Artificial  
<img width="1183" height="407" alt="ARQUITECTURA VISION" src="https://github.com/user-attachments/assets/78f47308-047e-4eb1-bfd8-fb4b39495f5b" />
**Fuente:** Proceso de proyectos Visión por Computador / FUENTE: Machine-Vision-Systems-Design A3 VISION

## 🧭 Flujo de Proceso para el Diseño e Implementación de un Proyecto de Visión por Computador

<img width="1414" height="854" alt="image" src="https://github.com/user-attachments/assets/ad4e652e-f2bd-4bf6-bdcd-1490adcbf762" />
**Fuente:** Proceso de proyectos Visión por Computador / FUENTE: Machine-Vision-Systems-Design A3 VISION

El siguiente flujo describe paso a paso el proceso típico para diseñar e implementar un sistema de **visión por computador (Machine Vision, MV)**.  
Acompaña al diagrama **“Typical Design Sequence”**, y sirve como guía práctica para la selección, integración y validación de los componentes ópticos, electrónicos y de software involucrados en un sistema industrial de visión.

---

## 1️⃣ Definir el tipo de sistema de visión (MV)
**Propósito:** elegir la arquitectura general (smart camera, cámara atada a controlador, PC-based).  
**Entradas:**
- Objetivos de calidad  
- Tasa de producción  
- Espacio disponible  
- Presupuesto  
- Integración con PLC o robot  

**Salidas:** arquitectura elegida, cantidad de cámaras, esquema de disparo (*trigger*) y sincronización.  

> 💡 **Nota de proyecto:** Si existen múltiples vistas o alta complejidad (mediciones precisas o uso de Machine Learning), suele ser más conveniente un sistema **PC-based** por su flexibilidad y capacidad de procesamiento.

---

## 2️⃣ Seleccionar la(s) cámara(s)
**Pasos:**
- Definir el **campo de visión (FOV)** con margen (10–20% de *overscan*).  
- Calcular la **resolución espacial requerida (mm/px)** a partir del tamaño mínimo a medir y la cantidad de píxeles necesarios para cubrir la característica.  
- Validar **velocidad** (fps y tiempo de exposición para que el desenfoque por movimiento sea ≤ 1 px).  
- Evaluar tipo de obturador (preferiblemente *global*), interfaz (GigE, USB3, CoaXPress).  

**Salidas:** modelo de cámara, resolución, fps, tipo de obturador e interfaz.

> 💡 **Nota de proyecto:** Si la línea se mueve rápido, priorizar *global shutter* y tiempos de exposición cortos.

---

## 3️⃣ Seleccionar el lente
**Pasos:**
- Calcular la **magnificación** (sensor/FOV) y la distancia de trabajo.  
- Seleccionar la **focal** que cumpla con el FOV deseado a la distancia real y asegure resolución óptica ≥ frecuencia de Nyquist del píxel.  
- Verificar **tipo de montura** (C/CS/S) y nivel de distorsión aceptable.  
- Considerar lentes **telecéntricos** cuando la precisión dimensional es crítica.  

**Salidas:** longitud focal, distancia de trabajo, tipo de montura y requisitos de resolución óptica.

---

## 4️⃣ Verificar cámara y lente
**Pruebas en banco:**
- Enfocar al FOV objetivo.  
- Medir la distancia de trabajo real.  
- Comprobar resolución en el centro y esquinas mediante una diana de calibración.  
- Revisar distorsión óptica.  

**Criterio de salida:** cumplimiento del FOV, nitidez y resolución en todo el campo visual.

---

## 5️⃣ Seleccionar la técnica/fuente de iluminación
**Pasos:**
- Partir de la **característica que requiere contraste** (bordes, alineación, huecos, textura, etc.).  
- Elegir **geometría óptica** (backlight, front light difusa, darkfield o brightfield).  
- Definir **difusividad, longitud de onda, polarización**, y modo de operación (continua o estroboscópica).  

**Salida:** modelo y geometría de iluminación + método de fijación mecánica.

> 💡 **Nota de proyecto:**  
> - Para contornos y *gaps*, un **backlight difuso** produce siluetas limpias.  
> - Para inspección de textura o superficie, se recomienda **front-light difusa**.

---

## 6️⃣ Verificar la imagen (Imaging)
**Montaje preliminar:**
- Capturar un lote representativo de muestras.  
- Evaluar exposición, uniformidad, contraste, reflejos, sombras y **SNR (Signal-to-Noise Ratio)**.  
- Ajustar ángulos, altura y ganancia.  

**Integrar el EDA de imágenes:**
- Histogramas de brillo y contraste.  
- Medición de desenfoque (varianza del Laplaciano).  
- Distribución de tamaños y balance de clases.  
- Detección de duplicados (*pHash*) y *outliers*.  

**Salidas:**  
- Parámetros finales de cámara/iluminación.  
- Plan de normalización (resize, recorte, normalización RGB).  
- Plan de *augmentation* según la variabilidad real observada.

---

## 7️⃣ Diseñar el procesamiento de imagen
**Definir el pipeline técnico** a partir del análisis EDA y los requisitos funcionales:

1. **Preprocesamiento:** redimensionamiento, normalización, reducción de ruido.  
2. **Localización/segmentación:** bordes, umbrales adaptativos, morfología, o modelo CNN/Detección.  
3. **Cálculos y mediciones:** alineación, separaciones, conteo, métricas geométricas.  
4. **Clasificación:** OK/Defecto o métricas de calidad.  

**Selección de herramientas:**
- Métodos clásicos (OpenCV, filtros, morfología).  
- Deep Learning (CNN, CNN+Transformers) según la separabilidad visual detectada (PCA/embeddings).

**Definir KPIs de desempeño:** precisión, recall, F1-score, tiempo de ciclo.  
**Protocolo de pruebas:** FAT (Factory Acceptance Test).  

**Salidas:**  
- Diagrama de bloques del pipeline.  
- Parámetros iniciales de procesamiento.  
- Dataset curado y dividido (train / val / test).

---

## 8️⃣ Implementar
**Actividades:**
- Integración con PLC, robot o HMI (protocolos EtherNet/IP, Profinet, etc.).  
- Calibraciones: intrínseca y mano-ojo si hay medidas en mm.  
- Construcción mecánica, cableado, seguridad y receta de producto.  
- Ejecución de **pruebas FAT** en taller con criterios definidos:  
  tasas de falsos ± máximas, tiempo de ciclo, estabilidad y repetibilidad.

---

## 9️⃣ Desplegar
**Fase de puesta en marcha:**
- Instalación y **SAT (Site Acceptance Test)** en planta.  
- Validación en condiciones reales de producción.  
- Capacitación a operadores y personal de mantenimiento.  
- Entrega de documentación técnica:  
  parámetros de configuración, planos eléctricos/mecánicos, *backups* y manuales.  

**Plan de soporte y mejora continua:**
- Registro de imágenes y logs de inspección.  
- Reentrenos periódicos y ajuste de umbrales adaptativos.  
- Mantenimiento de la iluminación, cámaras y óptica.

---

✅ **Resumen:**  
El proceso completo abarca desde la definición de la arquitectura hasta el despliegue en planta, asegurando que el sistema de visión cumpla los requisitos técnicos, productivos y de calidad bajo condiciones reales de operación industrial.


### Tipo de modelo utilizado y justificación
Se seleccionó una **ResNet-18** con *transfer learning* desde **ImageNet**, por su equilibrio entre precisión, velocidad y eficiencia en entornos industriales.  
Se reemplazó la capa final (`fc`) por una densa binaria para clasificación y se aplicó *dropout (0.3)* para regularización.

<img width="975" height="373" alt="image" src="https://github.com/user-attachments/assets/62d7545b-96d5-48e5-9c09-00436556dec0" />
**Fuente:** [CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)

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
