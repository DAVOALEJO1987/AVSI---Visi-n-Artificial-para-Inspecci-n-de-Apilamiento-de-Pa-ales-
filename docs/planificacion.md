# 🧭 PLANIFICACIÓN DEL PROYECTO AVSI
**Artificial Vision Stacking Inspection (AVSI)**  
*Sistema de visión artificial basado en machine learning para la inspección automática del apilamiento de pañales.*

---

## 🧩 1. Definición del Problema
La inspección manual del apilamiento de pañales en líneas de empaque representa un proceso crítico, sujeto a errores humanos, riesgos ergonómicos y variaciones en la calidad visual.  
El proyecto AVSI propone desarrollar un **prototipo funcional de visión artificial**, capaz de **detectar apilamientos incorrectos antes del empaque secundario**, utilizando una **red neuronal convolucional (CNN)** entrenada con datasets etiquetados.  
La solución integra **cámara industrial, preprocesamiento de imágenes y clasificación automática**, garantizando velocidad, precisión y trazabilidad en tiempo real.

---

## 🎯 2. Objetivos

### **Objetivo General**
Desarrollar en 6 semanas un **prototipo funcional de visión artificial basado en CNN** que verifique el apilamiento de pañales antes del empaque, alcanzando un **mAP ≥ 0.70**, **IoU > 0.50** y **>30 FPS**, reduciendo un **70% la intervención manual**.

### **Objetivos Específicos**
- Construir y etiquetar un dataset de al menos **100 imágenes** balanceadas (correcto/incorrecto).  
- Entrenar y validar una **CNN** alcanzando **precisión ≥ 90%** y **F1 ≥ 0.89**.  
- Implementar un módulo de procesamiento en tiempo real con cámara industrial.  
- Generar métricas reproducibles (mAP, IoU, FPS) en entorno controlado.  

---

## 💡 3. Justificación y Relevancia
La automatización de la inspección visual es una **necesidad emergente en la industria higiénica**.  
El proyecto representa una **innovación pionera** en el control de calidad de apilamiento de pañales, un área con vacíos de investigación.  
Su relevancia radica en:
- Reducir riesgos físicos y errores humanos.  
- Aumentar eficiencia y trazabilidad de procesos.  
- Validar el potencial de **Deep Learning** en entornos industriales con datasets pequeños.  
- Conectar la investigación académica con aplicaciones reales de la **Industria 4.0**.

---

## 📦 4. Alcance

### **Incluido**
- Captura, etiquetado y preprocesamiento de 100 imágenes.  
- Implementación de modelo CNN (ResNet-18) con data augmentation.  
- Validación en entorno controlado (laboratorio o simulador).  
- Documentación técnica, métricas y prototipo funcional.  

### **Excluido**
- Implementación en planta real o integración con PLC/SCADA.  
- Medición de impacto operativo o reducción de personal.  
- Escalamiento industrial del modelo (fase futura).  

---

## 🗓️ 5. Cronograma (Metodología Ágil – Scrum)

| Sprint | Semana | Objetivo Principal | Entregables | Riesgo | Mitigación |
|---------|---------|------------------|-------------|---------|-------------|
| **1** | 1–1.5 | Investigación y diseño | Documento de arquitectura, dataset inicial (100 imágenes) | Dataset insuficiente | Aplicar *data augmentation* |
| **2** | 2–3 | Desarrollo Core | Modelo CNN entrenado, pruebas unitarias | Bajo rendimiento | Ajustar capas, aumentar datos sintéticos |
| **3** | 4–5 | Optimización y validación | CNN optimizada (F1 ≥ 0.82), métricas y prototipo funcional | Overfitting | Regularización y validación cruzada |
| **4** | 6 | Cierre y documentación | Pruebas integrales, presentación final | Fallas finales | Pruebas incrementales desde semana 5 |

<img width="2375" height="1167" alt="Cronograma_Gantt_AVSI" src="https://github.com/user-attachments/assets/554bf4ee-a4e6-4916-84d9-5516fa9c18ce" />


---

## ⚙️ 6. Recursos Necesarios

### **Recursos Humanos**
| Rol | Función | Institución |
|------|----------|--------------|
| **Product Owner** | Supervisión académica, validación de entregables | UEES |
| **Scrum Team** | Desarrollo técnico (IA, dataset, código) | Estudiantes de Maestría |
| **Stakeholder** | Apoyo técnico-industrial, provisión de hardware | EVA ENGINEERING S.A. |

### **Recursos Técnicos**
| Tipo | Descripción | Cantidad |
|------|--------------|-----------|
| Hardware | Raspberry Pi 5, 8GB RAM, 128GB | 1 |
| Cámara industrial | SVPRO 4K 60fps, lente 2.8–12mm | 1 |
| Cables y materiales | Ethernet, montaje, alimentación | Varios |
| Software | Python, Visual Studio, TIA Portal, librerías IA | Licencias disponibles |

### **Recursos Financieros**
Auspicio directo de **EVA ENGINEERING S.A.**, que proporciona hardware, infraestructura, soporte técnico y dataset interno (código IN34-EVA1200 REV 01).

---

## ⚠️ 7. Riesgos Identificados y Estrategias de Mitigación

| Riesgo | Descripción | Mitigación |
|---------|--------------|-------------|
| Dataset limitado | Pocas imágenes reales para generalizar | *Data augmentation*, transfer learning |
| Baja precisión inicial | Modelo no converge adecuadamente | Ajuste de hiperparámetros, regularización |
| Overfitting | Dataset pequeño y modelo complejo | Early stopping, dropout, validación cruzada |
| Resistencia industrial | Temor a automatización laboral | Comunicación y capacitación técnica |
| Falta de pruebas reales | PoC no implementado en planta | Validación en entorno controlado y simulaciones |

---

## 🔍 8. Resultado Esperado
Un **prototipo validado** de inspección visual automatizada, con métricas reproducibles y arquitectura documentada, demostrando la **viabilidad técnica y escalabilidad industrial** de un sistema de visión artificial para la verificación de apilamiento.

---

## 📎 9. Referencias
Basado en el documento “**Presentación de Proyecto Integrador – Visión Artificial de Apilamiento de Pañales**”  
Materia: Proyecto Integrador en Inteligencia Artificial – MIAR0545  
Profesor: **PhD. Gladys Villegas**  
Alumnos: **Francisco Estupiñán & David Alejandro Narváez Mejía**  
Fecha: **19/09/2025**

