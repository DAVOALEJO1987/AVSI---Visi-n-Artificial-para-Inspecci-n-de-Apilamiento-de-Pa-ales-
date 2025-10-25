# 🧠 Manual de Usuario  
## Sistema AVSI – Visión Artificial de Inspección  

---

## 1. Descripción General del Sistema

El **Sistema AVSI (Artificial Vision for Stacking Inspection)** es una aplicación desarrollada en **Streamlit** que permite **clasificar el estado del apilamiento de pañales** mediante un modelo de **Visión por Computador (CNN)**.  
Su propósito es **detectar automáticamente errores de apilamiento** en procesos industriales de empaque, optimizando el control de calidad y reduciendo la intervención humana.

<img width="1620" height="1080" alt="image" src="https://github.com/user-attachments/assets/31898d0d-9ddd-438f-a6a5-2b51c48ab2db" />
**fuente** 

---

## 2. Requisitos del Sistema

| Requisito | Descripción |
|------------|-------------|
| **Sistema Operativo** | Windows 10/11, macOS o Linux |
| **Navegador Compatible** | Google Chrome, Edge o Firefox |
| **Conectividad** | Internet estable |
| **Formatos admitidos** | `.jpg` o `.png` |
| **Tamaño máximo de archivo** | 200 MB |
| **Dependencias principales** | `streamlit`, `opencv-python`, `tensorflow`, `numpy`, `plotly` |

---

## 3. Estructura de la Interfaz

La interfaz se organiza en **dos secciones principales**:
1. **Entrada de Datos**
2. **Predicción del Modelo**

Además, presenta una **barra superior de navegación** con pestañas informativas.

---

### 🔹 Encabezado

Incluye los logotipos institucionales de **UEES** y la **Maestría en Inteligencia Artificial**, junto con el título:

> **“Visión Artificial de Inspección – AVSI”**

Identifica la aplicación como parte del proyecto académico de IA aplicada a procesos industriales.

---

### 🔹 Pestañas Superiores

- **Funcionamiento:** muestra la interfaz principal y flujo de trabajo del modelo.  
- **Información:** incluye detalles técnicos, métricas de desempeño y recomendaciones de uso.

---

## 4. Sección 1 – Entrada de Datos

Esta sección permite cargar la imagen que será analizada por el modelo.

### Opciones disponibles:

1. **Subir Archivo:**  
   - Haga clic en “**Browse files**” o arrastre la imagen al área **Drag and drop file here**.  
   - Se admiten archivos `.jpg` o `.png`.

2. **Tomar Foto (Webcam):**  
   - Activa la cámara del computador para capturar imágenes en tiempo real.  
   - Ideal para pruebas en planta o entorno académico.

3. **Usar Ejemplo:**  
   - Carga una imagen de muestra predefinida para demostración.

> 💡 **Consejo:** use imágenes bien iluminadas, con foco y fondo uniforme para obtener una predicción precisa.

---

## 5. Sección 2 – Predicción del Modelo

Una vez cargada la imagen, el sistema activa los controles de inferencia.

### **Botón: “Ejecutar Predicción”**
- Inicia el proceso de clasificación.  
- El modelo analiza la imagen, extrae características visuales (color, forma, textura) y determina la categoría correspondiente.

---

### **Resultado de la Clasificación**

El sistema presenta el diagnóstico junto al porcentaje de confianza del modelo.

| Campo | Descripción |
|--------|--------------|
| **Predicción** | Resultado del análisis (`MAL APILADO` o `BIEN APILADO`) |
| **Nivel de Confianza (%)** | Porcentaje de probabilidad asignado por la red neuronal |

📊 **Ejemplo mostrado:**
> Predicción: **MAL APILADO**  
> Nivel de Confianza: **78.79 %**

---

## 6. Interpretación de Resultados

| Resultado | Significado | Acción sugerida |
|------------|--------------|----------------|
| **BIEN APILADO** | Cumple con los estándares de calidad. | No requiere intervención. |
| **MAL APILADO** | Desviaciones en altura o alineación de la pila. | Revisar físicamente o detener la línea para inspección. |

---

## 7. Sección de Ayuda

El botón **“Ayuda”** despliega orientación sobre el uso del sistema, recomendaciones de imágenes y solución de errores comunes.

---

## 8. Buenas Prácticas de Uso

- Utilizar imágenes en buena iluminación y ángulo frontal.  
- Evitar fondos confusos o con objetos adicionales.  
- Actualizar el modelo periódicamente para mejorar la precisión.  
- Entrenar el modelo con datasets balanceados y correctamente etiquetados.

---

## 9. Posibles Errores y Soluciones

| Error | Causa | Solución |
|-------|--------|----------|
| “Formato no admitido” | Archivo diferente a `.jpg` o `.png` | Convertir la imagen al formato correcto. |
| “Archivo demasiado grande” | Imagen mayor a 200 MB | Reducir tamaño o resolución. |
| “Predicción no disponible” | Error de conexión o modelo | Recargar la página y repetir
