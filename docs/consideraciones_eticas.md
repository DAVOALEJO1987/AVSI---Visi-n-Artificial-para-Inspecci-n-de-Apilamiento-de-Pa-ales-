# ⚖️ Consideraciones Éticas del Proyecto AVSI

Este documento presenta un análisis exhaustivo de las **consideraciones éticas** inherentes al diseño, despliegue y operación del sistema **AVSI (Artificial Vision Stacking Inspection)**.  
El análisis se alinea con los principios de **IA Responsable (Responsible AI)**, evaluando los riesgos y proponiendo estrategias de mitigación desde una perspectiva sociotécnica.

---

## 1️⃣ Análisis de Sesgos y Generalización

En el contexto de AVSI, el análisis de sesgos se enfoca en los **sesgos técnicos y de representación**, que afectan la robustez y fiabilidad del modelo.

### 🔹 Sesgo de Representación
El dataset base fue capturado bajo un conjunto limitado de condiciones (iluminación, planta, configuración de cámara).  
Esto genera un **sesgo de muestreo** que limita la capacidad de **generalización** del modelo ante entornos distintos, fenómeno conocido como *domain shift*.

### 🔹 Sesgo de Medición
Diferencias sistemáticas entre los datos de entrenamiento y el entorno real (calibración de cámara, compresión de video, desenfoque por movimiento) introducen un **sesgo de medición** que puede degradar el rendimiento predictivo.

### 🔹 Impacto Predictivo
Estos sesgos se manifiestan como un aumento de **falsos positivos (FPR)** o **falsos negativos (FNR)**.  
Por ejemplo, un modelo entrenado solo con buena iluminación podría fallar durante turnos nocturnos, reduciendo su confiabilidad.

### 🔹 Grupos Afectados
- Exceso de falsos positivos → fatiga por alertas y sobrecarga cognitiva en operarios.  
- Exceso de falsos negativos → riesgos para el consumidor y evaluaciones injustas del personal de calidad.

---

## 2️⃣ Equidad y Fairness Operativo

Dado que AVSI inspecciona objetos, la **equidad** se redefine como **consistencia operativa del rendimiento**.

### 🔹 Definición de Equidad Operativa
Un sistema equitativo mantiene métricas estables (precisión, F1-score) sin importar turno, línea, lote o supervisor.

### 🔹 Métricas de Evaluación
Se auditan métricas de error (FPR, FNR) de forma desagregada por variables operativas.  
Diferencias notables entre turnos o líneas indican inequidad operativa.

### 🔹 Estrategias de Mitigación
- **Data Augmentation:** simular variabilidad de dominio (brillo, contraste, ruido).  
- **Muestreo Estratificado:** asegurar representación de condiciones minoritarias o complejas.

---

## 3️⃣ Privacidad y Gobernanza de Datos

Los sistemas de visión industrial pueden captar datos contextuales del entorno de trabajo, implicando consideraciones de **privacidad laboral**.

### 🔹 Datos Sensibles Contextuales
Aunque el objetivo es el producto, las cámaras pueden capturar indirectamente información sobre los operarios (movimientos, posiciones, hábitos).

### 🔹 Mitigación bajo “Privacy by Design”
- **Técnica (Minimización de Datos):** recorte (*cropping*) o desenfoque de zonas periféricas antes de almacenar imágenes.  
- **Gobernanza (Limitación de Propósito):** uso exclusivo para inspección de calidad.  
  - Acceso restringido al equipo de MLOps.  
  - Prohibido para fines de supervisión o recursos humanos.

---

## 4️⃣ Transparencia y Explicabilidad (XAI)

El modelo ResNet-18 ofrece alto rendimiento, pero su naturaleza de **caja negra** requiere estrategias de **explicabilidad**.

### 🔹 Transparencia
A nivel de interfaz, la app **Streamlit** muestra inferencias en tiempo real, superponiendo la clasificación (“Correcto” / “Incorrecto”) y permitiendo al operario ver qué analiza el sistema.

### 🔹 Explicabilidad (Explainable AI)
Durante validación y auditoría, se aplican técnicas *post-hoc* como **Grad-CAM**, que visualizan las zonas relevantes de la imagen usadas por el modelo.  
Esto permite confirmar que las decisiones se basan en características del producto y no en artefactos del fondo.

---

## 5️⃣ Análisis de Impacto Social (Stakeholders)

La introducción de AVSI impacta de forma diferente a cada grupo involucrado en la cadena de producción.

### 🔹 Impactos Positivos
| Grupo | Impacto |
|--------|----------|
| **Empresa** | Eficiencia productiva, reducción de mermas, mejora de calidad. |
| **Consumidor** | Mayor garantía de calidad en el producto final. |
| **Operarios** | Reducción de tareas repetitivas y riesgos ergonómicos. |

### 🔹 Impactos Negativos / Riesgos Sociotécnicos
- **Automatización de tareas:** requiere **reentrenamiento** del personal (de inspector a supervisor de IA).  
- **Ansiedad por vigilancia:** percepción de monitoreo constante que puede afectar el clima laboral.  
- **Complacencia automatizada:** exceso de confianza en el sistema, reduciendo la vigilancia humana ante posibles errores.

---

## 6️⃣ Responsabilidad y “Human-in-the-Loop” (HITL)

La responsabilidad se distribuye entre el equipo técnico y la gestión operativa.

### 🔹 Asignación de Responsabilidad
- **Equipo de IA:** validación, robustez, documentación.  
- **Gerencia de planta:** implementación, protocolos y capacitación.

### 🔹 Mecanismos de Accountability
- **Human-in-the-Loop (HITL):** el operario supervisa el sistema y valida las alertas.  
- **MLOps y Gobernanza:** monitoreo continuo del modelo (detección de *model drift*) y retroalimentación para reentrenamiento con datos curados.

---

## 7️⃣ Uso Dual y Riesgos Ético-Laborales

El riesgo de “uso dual” se refiere al **empleo indebido** del sistema con fines no previstos.

### 🔹 Riesgo de Mal Uso
El sistema podría usarse para **vigilar la productividad de los empleados**, lo cual constituiría una violación ética.

### 🔹 Estrategias de Salvaguarda
- **Controles Técnicos:** anonimización y privacidad por defecto.  
- **Controles de Gobernanza:** políticas internas que prohíben el uso del sistema para evaluación del desempeño personal.

---

## 8️⃣ Limitaciones Reconocidas y Robustez

### 🔹 Datos Fuera de Distribución (OOD)
El modelo no es confiable para productos o empaques distintos a los del entrenamiento.  
Se requiere reentrenamiento o validación antes de cambios de diseño.

### 🔹 Perturbaciones del Dominio
El rendimiento puede degradarse con variaciones ambientales (iluminación, sombras, velocidad de cinta).

### 🔹 Casos Límite
- Oclusiones parciales (mano u objeto tapando el producto).  
- Presentaciones anómalas del paquete (giro, posición lateral).

### ⚠️ Advertencia de Uso
El sistema AVSI es una herramienta de **asistencia a la inspección**, no un reemplazo total del juicio humano.  
La supervisión humana cualificada sigue siendo indispensable para garantizar la calidad del producto.

---

## 📚 Referencia de Principios Éticos

1. **IA Responsable (RAI)** – IEEE Global Initiative on Ethics of Autonomous Systems.  
2. **Privacy by Design** – Cavoukian, A. (2010).  
3. **Human-in-the-Loop Frameworks** – Amershi et al. (2019).  
4. **XAI – Explainable Artificial Intelligence** – DARPA Initiative (2018).  
5. **Ethical AI in Industrial Automation** – OECD AI Principles (2021).

<img width="930" height="1395" alt="image" src="https://github.com/user-attachments/assets/025e2581-8230-4e3f-9466-d406a3e4ac2d" />

**Fuente:** Generado por ChatGPT
