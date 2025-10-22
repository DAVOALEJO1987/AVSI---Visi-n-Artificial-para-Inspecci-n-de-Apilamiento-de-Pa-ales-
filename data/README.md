# 📂 Carpeta de Datos – Proyecto AVSI
**Artificial Vision Stacking Inspection (AVSI)**  
Sistema de Visión por Computador e Inteligencia Artificial para la inspección automática del apilamiento de pañales en líneas de empaque industrial.  

---

## 📊 Estructura General

---

## 🧩 1. Dataset Original (`/data/raw/`)

### 🔹 **Dataset 1 – 100 Imágenes**
- **Procedencia:** EVA ENGINEERING S.A.  
- **Código de autorización:** IN34-EVA1200 REV 01  
- **Licencia:** Uso académico y prototipado interno.  
- **Contenido:** 100 imágenes capturadas en planta industrial bajo distintas condiciones de iluminación y orientación.  
- **Propósito:** Validar la arquitectura inicial del modelo (baseline con ResNet-18).  

### 🔹 **Dataset 2 – 1 000 Imágenes**
- **Procedencia:** Ampliación interna mediante recopilación y técnicas de *data augmentation*.  
- **Contenido:** Imágenes rotadas, reflejadas, y con variaciones de brillo, contraste y posición.  
- **Propósito:** Entrenamiento final del modelo optimizado y validación del impacto de los datos en el rendimiento.  
- **Resultado:** Mejora de mAP ≥ 0.70 e IoU > 0.50, con rendimiento en tiempo real > 30 FPS.

---

## ⚙️ 2. Datos Procesados (`/data/processed/`)

Los datos procesados se generan a partir de los conjuntos *raw* aplicando:

- Redimensionamiento uniforme (224×224 píxeles).  
- Normalización de color RGB.  
- Filtrado de ruido y duplicados.  
- Balanceo de clases mediante **SMOTE**.  
- División estratificada en **train**, **validation** y **test**.  

Ejemplo de estructura:

---

## 🧠 3. Política de Uso y Ética

- Los datos pertenecen a **EVA ENGINEERING S.A.** y se utilizan con fines de investigación académica.  
- Se prohíbe su distribución fuera del contexto educativo del proyecto AVSI.  
- Todos los datasets cumplen principios de **ética y privacidad industrial**, sin contener información sensible ni personal.

---

## 🧾 4. Archivo `labels.csv`

El archivo `labels.csv` describe la correspondencia entre nombre de imagen y clase asignada.  
Formato:

| filename | class       |
|-----------|-------------|
| img_001.jpg | good_stack |
| img_002.jpg | bad_stack  |

---

## 📦 5. Reproducibilidad

Para regenerar los datos procesados desde los crudos:

```bash
python src/data_processing.py


---
