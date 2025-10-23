
import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from pathlib import Path
import pandas as pd
import io
import numpy as np
from datetime import datetime
import time
import plotly.express as px
from fpdf import FPDF
import base64
import pytz

# --- 0. CONFIGURACIÓN DE IDIOMA E HISTORIAL ---
if 'lang' not in st.session_state: st.session_state.lang = 'es'
if 'history' not in st.session_state: st.session_state.history = []
if 'load_example' not in st.session_state: st.session_state.load_example = False
if 'file_error' not in st.session_state: st.session_state.file_error = None
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'camera_key' not in st.session_state: st.session_state.camera_key = 1000

translations = {
    'es': {
        'lang_select': 'Seleccionar Idioma',
        'page_title': 'Clasificador de Apilado',
        'info': 'Sube una imagen, usa tu cámara web o carga un ejemplo para clasificar.',
        'col1_header': '1. Entrada de Datos',
        'tab1_label': 'Subir Archivo',
        'tab1_caption': 'Sube un archivo JPG o PNG desde tu PC.',
        'tab1_uploader': 'Carga una imagen',
        'tab2_label': 'Tomar Foto (Webcam)',
        'tab2_caption': "Permite el acceso a la cámara. Haz clic en 'Take Photo' para capturar.",
        'tab2_uploader': 'Captura desde la webcam',
        'tab2_clear_caption': "Usa el botón 'Clear photo' para borrar la foto.",
        'tab3_label': 'Usar Ejemplo',
        'tab3_caption': 'Probar el modelo con una imagen de ejemplo.',
        'tab3_button': 'Cargar Imagen de Ejemplo',
        'img_caption': 'Imagen seleccionada',
        'col2_header': '2. Predicción del Modelo',
        'run_button': 'Ejecutar Predicción',
        'progress_bar_text': 'Ejecutando predicción...',
        'pred_header': 'Resultado de la Clasificación',
        'pred_label': 'Predicción:',
        'conf_label': 'Nivel de Confianza',
        'probs_header': 'Probabilidades por Clase',
        'info_wait': "Selecciona una imagen y presiona 'Ejecutar Predicción' para ver los resultados.",
        'bien_apilado': 'BIEN APILADO',
        'mal_apilado': 'MAL APILADO',
        'class_fallback': 'Clase:',
        'history_header': 'Historial de Predicciones (Sesión)',
        'chart_header': 'Comparativa de Confianza',
        'chart_title': 'Confianza de Predicciones en la Sesión',
        'chart_y_axis': 'Nivel de Confianza',
        'chart_x_axis': 'Nro. de Predicción',
        'export_header': 'Exportar Historial',
        'export_csv_button': 'Descargar como .CSV',
        'export_pdf_button': 'Descargar como .PDF',
        'col_pred_num': 'Nro.',
        'col_timestamp': 'Fecha/Hora',
        'col_source': 'Fuente',
        'col_prediction': 'Predicción',
        'col_confidence': 'Confianza (%)',
        'source_upload': 'Archivo Subido',
        'source_webcam': 'Webcam',
        'source_example': 'Ejemplo',
        'help_popover': '❓ Ayuda',
        'help_step2_verify': 'Espera a que la imagen aparezca cargada en la columna de la izquierda.',
        'help_step3_run': "Ve a la columna '2. Predicción del Modelo' y haz clic en el botón 'Ejecutar Predicción'.",
        'help_upload_title': 'Cómo Predecir desde un Archivo',
        'help_upload_step1_title': 'Paso 1: Cargar la Imagen',
        'help_upload_step1_desc': "Haz clic en 'Browse files' o arrastra una imagen (JPG, PNG) a la zona de carga.",
        'help_webcam_title': 'Cómo Predecir desde la Webcam',
        'help_webcam_step1_title': 'Paso 1: Capturar Foto',
        'help_webcam_step1_desc': "Permite el acceso a la cámara y haz clic en 'Take photo' para capturar una imagen.",
        'help_webcam_step2_clear': "Si necesitas otra foto, haz clic en 'Clear photo' para borrar la actual.",
        'help_example_title': 'Cómo Predecir con una Imagen de Ejemplo',
        'help_example_step1_title': 'Paso 1: Cargar Ejemplo',
        'help_example_step1_desc': "Haz clic en 'Cargar Imagen de Ejemplo' para cargar una imagen predefinida.",
        'error_model_load': 'Error Fatal: No se pudo cargar el archivo del modelo desde la ruta: ',
        'error_example_load': 'Error: No se pudo cargar la imagen de ejemplo desde la ruta: ',
        'error_processing': 'Error de Predicción: Ocurrió un problema al procesar la imagen.',
        'error_file_type': 'Error: El archivo no es una imagen (JPG, PNG). Por favor, carga un archivo compatible.',
        'error_header_load': 'Error: No se pudo cargar la imagen del encabezado desde la ruta: ',
        'error_presentation_load': 'Error: No se pudo cargar la imagen de presentación desde la ruta: ',
        'error_description_load': 'Error: No se pudo cargar la imagen de descripción desde la ruta: ',
        'error_architecture_load': 'Error: No se pudo cargar la imagen de arquitectura desde la ruta: ',
        'main_tab_function': 'Funcionamiento',
        'main_tab_info': 'Información',
        'info_expander_project': 'Proyecto',
        'info_expander_description': 'Descripción',
        'info_expander_architecture': 'Arquitectura',
    },
    'en': {
        'lang_select': 'Select Language',
        'page_title': 'Stacking Classifier',
        'info': 'Upload an image, use your webcam, or load an example to classify.',
        'col1_header': '1. Data Input',
        'tab1_label': 'Upload File',
        'tab1_caption': 'Upload a JPG or PNG file from your PC.',
        'tab1_uploader': 'Load an image',
        'tab2_label': 'Take Photo (Webcam)',
        'tab2_caption': "Allow camera access. Click 'Take Photo' to capture.",
        'tab2_uploader': 'Capture from webcam',
        'tab2_clear_caption': "Use the 'Clear photo' button to clear the picture.",
        'tab3_label': 'Use Example',
        'tab3_caption': 'Test the model with an example image.',
        'tab3_button': 'Load Example Image',
        'img_caption': 'Selected Image',
        'col2_header': '2. Model Prediction',
        'run_button': 'Run Prediction',
        'progress_bar_text': 'Running prediction...',
        'pred_header': 'Classification Result',
        'pred_label': 'Prediction:',
        'conf_label': 'Confidence Level',
        'probs_header': 'Probabilities per Class',
        'info_wait': "Select an image and press 'Run Prediction' to see the results.",
        'bien_apilado': 'GOOD STACK',
        'mal_apilado': 'BAD STACK',
        'class_fallback': 'Class:',
        'history_header': 'Prediction History (Session)',
        'chart_header': 'Confidence Comparison',
        'chart_title': 'Prediction Confidence per Session',
        'chart_y_axis': 'Confidence Level',
        'chart_x_axis': 'Prediction #',
        'export_header': 'Export History',
        'export_csv_button': 'Download as .CSV',
        'export_pdf_button': 'Download as .PDF',
        'col_pred_num': '#',
        'col_timestamp': 'Timestamp',
        'col_source': 'Source',
        'col_prediction': 'Prediction',
        'col_confidence': 'Confidence (%)',
        'source_upload': 'File Upload',
        'source_webcam': 'Webcam',
        'source_example': 'Example',
        'help_popover': '❓ Help',
        'help_step2_verify': 'Wait for the image to appear loaded in the left column.',
        'help_step3_run': "Go to the '2. Model Prediction' column and click the 'Run Prediction' button.",
        'help_upload_title': 'How to Predict from a File',
        'help_upload_step1_title': 'Step 1: Upload the Image',
        'help_upload_step1_desc': "Click 'Browse files' or drag and drop an image (JPG, PNG) into the upload area.",
        'help_webcam_title': 'How to Predict from Webcam',
        'help_webcam_step1_title': 'Step 1: Capture Photo',
        'help_webcam_step1_desc': "Allow camera access and click 'Take photo' to capture an image.",
        'help_webcam_step2_clear': "If you need another shot, click 'Clear photo' to delete the current one.",
        'help_example_title': 'How to Predict with an Example Image',
        'help_example_step1_title': 'Step 1: Load Example',
        'help_example_step1_desc': "Click 'Load Example Image' to load a predefined image.",
        'error_model_load': 'Fatal Error: Could not load the model file from path: ',
        'error_example_load': 'Error: Could not load the example image from path: ',
        'error_processing': 'Prediction Error: An problem occurred while processing the image.',
        'error_file_type': 'Error: The file is not an image (JPG, PNG). Please upload a compatible file.',
        'error_header_load': 'Error: Could not load the header image from path: ',
        'error_presentation_load': 'Error: Could not load the presentation image from path: ',
        'error_description_load': 'Error: Could not load the description image from path: ',
        'error_architecture_load': 'Error: Could not load the architecture image from path: ',
        'main_tab_function': 'Functionality',
        'main_tab_info': 'Information',
        'info_expander_project': 'Project',
        'info_expander_description': 'Description',
        'info_expander_architecture': 'Architecture',
    }
}

def t(key):
    # Try direct key lookup first
    if key in translations[st.session_state.lang]:
        return translations[st.session_state.lang][key]

    # Fallback for prediction classes
    name = str(key).lower()
    if name.startswith(("ok", "bien", "good")): return t('bien_apilado')
    if name.startswith(("ng", "mal", "bad")): return t('mal_apilado')

    # Fallback for source keys
    if key == 'source_upload': return t('source_upload')
    if key == 'source_webcam': return t('source_webcam')
    if key == 'source_example': return t('source_example')

    return key # Return original key if no translation found

# --- 1. CONFIGURACIÓN DE RUTAS ---
DRIVE_PATH = Path('/content/drive/MyDrive/')
MODEL_PATH = DRIVE_PATH / 'outputs/cnn_best.pt'
EXAMPLE_IMG_PATH = DRIVE_PATH / 'example/TEST_SET_MAL_0032.jpg'
# Base de nombres de archivo (se construirán dinámicamente)
HEADER_BASE_NAME = 'encabezado'; PRESENTATION_BASE_NAME = 'presentacion_proyecto'
DESCRIPTION_BASE_NAME = 'descripcion'; ARCHITECTURE_BASE_NAME = 'arquitectura'
IMG_EXT = '.png'

# --- 2. DEFINICIÓN DEL MODELO ---
def build_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, num_classes))
    return model

# --- 3. FUNCIONES DE AYUDA ---
infer_tfms = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model_and_mappings(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not model_path.exists():
        st.error(f"{t('error_model_load')} {model_path}"); return None, None, None
    try:
        ckpt = torch.load(str(model_path), map_location=device)
        num_classes = len(ckpt["class_to_idx"])
        model = build_model(num_classes=num_classes)
        model.load_state_dict(ckpt["state_dict"]); model.eval().to(device)
        IDX2CLS = {v: k for k, v in ckpt["class_to_idx"].items()}
        return model, IDX2CLS, device
    except Exception as e:
        st.error(f"{t('error_model_load')} {model_path}. Error: {e}"); return None, None, None

def clasificar_imagen(model, img_pil, idx2cls, device):
    img = ImageOps.exif_transpose(img_pil)
    x = infer_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x); probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs)); pred_cls = idx2cls[pred_idx]
    etiqueta = t(pred_cls); confianza = float(probs[pred_idx])
    all_probs_data = {t(idx2cls[i]): probs[i] for i in range(len(probs))}
    df_probs = pd.DataFrame(all_probs_data.items(), columns=[t('class_fallback'), t('col_confidence')])
    df_probs = df_probs.sort_values(by=t('col_confidence'), ascending=False).reset_index(drop=True)
    return etiqueta, confianza, df_probs, pred_cls

class PDF(FPDF):
    def header(self): self.set_font('Arial', 'B', 12); self.cell(0, 10, t('history_header'), 0, 1, 'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def add_table_row(self, data, is_header=False):
        font = ('Arial', 'B', 10) if is_header else ('Arial', '', 10); self.set_font(*font)
        col_widths = [15, 45, 30, 60, 40]
        for i, item in enumerate(data): self.cell(col_widths[i], 10, str(item).encode('latin-1', 'replace').decode('latin-1'), 1)
        self.ln()

def create_pdf(history_df):
    pdf = PDF(); pdf.add_page()
    headers = [t('col_pred_num'), t('col_timestamp'), t('col_source'), t('col_prediction'), t('col_confidence')]
    pdf.add_table_row(headers, is_header=True)
    for idx, row in history_df.iterrows():
        data = [row[h] for h in headers]
        pdf.add_table_row(data)
    return bytes(pdf.output(dest='S'))

@st.cache_data
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def clear_for_uploader():
    st.session_state.file_error = None; st.session_state.load_example = False
    st.session_state.camera_key += 1
def clear_for_camera():
    st.session_state.file_error = None; st.session_state.load_example = False
    st.session_state.uploader_key += 1
def click_example_button():
    st.session_state.file_error = None; st.session_state.load_example = True
    st.session_state.uploader_key += 1; st.session_state.camera_key += 1
def clear_history_on_lang_change(): st.session_state.history = []

# --- 4. CONSTRUCCIÓN DE LA INTERFAZ ---
st.set_page_config(layout="wide", page_title=t('page_title'))

# CSS
st.markdown(
    """
    <style>
    .fixed-header {
        position: fixed; top: 0; left: 0; width: 100%; z-index: 1000;
        background-color: white; padding: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fixed-header img { width: 100%; height: auto; display: block; }
    .block-container {
        padding-top: 150px; /* <<< AJUSTA ESTO a la altura de tu encabezado.png */
    }
    div[data-testid="stTabs"] > div[role="tablist"] > button[role="tab"] p {
        font-size: 1.25rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.selectbox(t('lang_select'), options=['es', 'en'], key='lang', on_change=clear_history_on_lang_change)

# Cargar Modelo
model, IDX2CLS, device = load_model_and_mappings(MODEL_PATH)

# Construir rutas dinámicas (después del idioma)
LANG_SUFFIX = '_en' if st.session_state.lang == 'en' else ''
HEADER_IMG_PATH = DRIVE_PATH / f'source/{HEADER_BASE_NAME}{LANG_SUFFIX}{IMG_EXT}'
PRESENTATION_IMG_PATH = DRIVE_PATH / f'source/{PRESENTATION_BASE_NAME}{LANG_SUFFIX}{IMG_EXT}'
DESCRIPTION_IMG_PATH = DRIVE_PATH / f'source/{DESCRIPTION_BASE_NAME}{LANG_SUFFIX}{IMG_EXT}'
ARCHITECTURE_IMG_PATH = DRIVE_PATH / f'source/{ARCHITECTURE_BASE_NAME}{LANG_SUFFIX}{IMG_EXT}'

# Header
with st.container():
    st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
    if HEADER_IMG_PATH.exists():
        try: st.image(str(HEADER_IMG_PATH), use_container_width=True)
        except Exception as e: st.error(f"{t('error_header_load')} {HEADER_IMG_PATH}. Error: {e}")
    else: st.error(f"{t('error_header_load')} {HEADER_IMG_PATH}")
    st.markdown('</div>', unsafe_allow_html=True)

# Pestañas Principales
main_tab1, main_tab2 = st.tabs([t('main_tab_function'), t('main_tab_info')])

# --- Pestaña de Funcionamiento ---
with main_tab1:
    if model is None: st.warning("El modelo no se cargó correctamente. Revisa los errores y la ruta del modelo.")
    else:
        st.info(t('info'))
        img_a_procesar = None; raw_source_key = None
        col1, col2 = st.columns(2)
        with col1: # Input
             st.header(t('col1_header'))
             tab1, tab2, tab3 = st.tabs([t('tab1_label'), t('tab2_label'), t('tab3_label')])
             with tab1: # Upload
                 col_cap, col_pop = st.columns([0.8, 0.2])
                 with col_cap: st.caption(t('tab1_caption'))
                 with col_pop:
                     with st.popover(t('help_popover')):
                         st.subheader(t('help_upload_title')); st.markdown(f"**{t('help_upload_step1_title')}**"); st.markdown(t('help_upload_step1_desc'))
                         st.markdown(f"**{t('help_step2_verify')}**"); st.markdown(f"**{t('help_step3_run')}**")
                 if st.session_state.file_error: st.error(st.session_state.file_error)
                 uploaded_file = st.file_uploader(t('tab1_uploader'), type=None, label_visibility="collapsed", key=f"file_uploader_{st.session_state.uploader_key}", on_change=clear_for_uploader)
             with tab2: # Webcam
                 col_cap, col_pop = st.columns([0.8, 0.2])
                 with col_cap: st.caption(t('tab2_caption'))
                 with col_pop:
                     with st.popover(t('help_popover')):
                         st.subheader(t('help_webcam_title')); st.markdown(f"**{t('help_webcam_step1_title')}**"); st.markdown(t('help_webcam_step1_desc'))
                         st.markdown(f"**{t('help_webcam_step2_clear')}**"); st.markdown(f"**{t('help_step2_verify')}**"); st.markdown(f"**{t('help_step3_run')}**")
                 camera_img = st.camera_input(t('tab2_uploader'), label_visibility="collapsed", key=f"camera_{st.session_state.camera_key}", on_change=clear_for_camera)
                 st.caption(t('tab2_clear_caption'))
             with tab3: # Example
                 col_cap, col_pop = st.columns([0.8, 0.2])
                 with col_cap: st.caption(t('tab3_caption'))
                 with col_pop:
                     with st.popover(t('help_popover')):
                         st.subheader(t('help_example_title')); st.markdown(f"**{t('help_example_step1_title')}**"); st.markdown(t('help_example_step1_desc'))
                         st.markdown(f"**{t('help_step2_verify')}**"); st.markdown(f"**{t('help_step3_run')}**")
                 st.button(t('tab3_button'), on_click=click_example_button)
             # Load image logic
             try:
                 if uploaded_file: img_a_procesar = Image.open(uploaded_file).convert("RGB"); raw_source_key = 'source_upload'
                 elif camera_img: img_a_procesar = Image.open(camera_img).convert("RGB"); raw_source_key = 'source_webcam'
                 elif st.session_state.load_example:
                     if EXAMPLE_IMG_PATH.exists(): img_a_procesar = Image.open(EXAMPLE_IMG_PATH).convert("RGB"); raw_source_key = 'source_example'
                     else: st.error(t('error_example_load')); st.session_state.load_example = False
             except Exception as e:
                 if uploaded_file: st.session_state.file_error = t('error_file_type'); st.session_state.uploader_key += 1; st.rerun()
                 else: st.error(f"{t('error_processing')}: {e}"); img_a_procesar = None
             if img_a_procesar: st.image(img_a_procesar, caption=t('img_caption'), use_container_width=True)

        with col2: # Output
            st.header(t('col2_header'))
            if img_a_procesar is not None and model is not None:
                if st.button(t('run_button')):
                    try:
                        progress_bar = st.progress(0, text=t('progress_bar_text'))
                        for percent_complete in range(100): time.sleep(0.01); progress_bar.progress(percent_complete + 1, text=t('progress_bar_text'))
                        etiqueta, confianza, df_probs, pred_cls = clasificar_imagen(model, img_a_procesar, IDX2CLS, device)
                        progress_bar.empty()
                        # Display results
                        st.subheader(t('pred_header'))
                        if "MAL" in etiqueta or "BAD" in etiqueta: st.error(f"**{t('pred_label')}** {etiqueta}")
                        else: st.success(f"**{t('pred_label')}** {etiqueta}")
                        st.metric(label=t('conf_label'), value=f"{confianza:.2%}")
                        st.subheader(t('probs_header')); st.dataframe(df_probs, use_container_width=True)
                        # Save to history
                        ecuador_tz=pytz.timezone('America/Guayaquil'); now_utc=datetime.now(pytz.utc); now_ecuador=now_utc.astimezone(ecuador_tz); timestamp_str=now_ecuador.strftime('%Y-%m-%d %H:%M:%S')
                        history_entry = {'timestamp': timestamp_str, 'source_key': raw_source_key, 'prediction_key': pred_cls, 'confidence': confianza}
                        st.session_state.history.append(history_entry)
                    except Exception as e:
                        if 'progress_bar' in locals(): progress_bar.empty()
                        st.error(f"{t('error_processing')}: {e}")
            else:
                 if model is not None: st.info(t('info_wait'))

        # Historial y Gráficos
        if st.session_state.history:
            st.divider(); st.header(t('history_header'))
            history_df = pd.DataFrame(st.session_state.history)
            df_display = history_df.copy()
            df_display[t('col_prediction')] = df_display['prediction_key'].apply(t); df_display[t('col_source')] = df_display['source_key'].apply(t)
            df_display[t('col_pred_num')] = range(1, len(df_display) + 1); df_display['confidence_str'] = df_display['confidence'].map('{:.2%}'.format)
            df_display.rename(columns={'timestamp': t('col_timestamp'), 'confidence_str': t('col_confidence')}, inplace=True)
            df_display_table = df_display[[t('col_pred_num'), t('col_timestamp'), t('col_source'), t('col_prediction'), t('col_confidence')]]
            st.dataframe(df_display_table, use_container_width=True)
            st.subheader(t('chart_header'))
            df_chart = df_display.copy(); df_chart[t('chart_x_axis')] = range(1, len(df_chart) + 1)
            color_map = {t('bien_apilado'): '#28a745', t('mal_apilado'): '#dc3545'}
            fig = px.bar(df_chart, x=t('chart_x_axis'), y='confidence', title=t('chart_title'), labels={'confidence': t('chart_y_axis')}, text=df_chart['confidence'].map('{:.1%}'.format), color=t('col_prediction'), color_discrete_map=color_map)
            fig.update_yaxes(range=[0, 1], tickformat=".0%"); fig.update_layout(legend_title_text=t('col_prediction'))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader(t('export_header'))
            df_export = df_display_table.copy(); col_export1, col_export2 = st.columns(2)
            with col_export1: csv_data = convert_df_to_csv(df_export); st.download_button(label=t('export_csv_button'), data=csv_data, file_name="historial_predicciones.csv", mime="text/csv")
            with col_export2: pdf_data = create_pdf(df_export); st.download_button(label=t('export_pdf_button'), data=pdf_data, file_name="historial_predicciones.pdf", mime="application/pdf")

# --- Pestaña de Información ---
with main_tab2:
    st.header(t('main_tab_info'))
    # Las rutas se reconstruyen aquí usando el LANG_SUFFIX actual
    current_presentation_path = DRIVE_PATH / f'source/{PRESENTATION_BASE_NAME}{LANG_SUFFIX}{IMG_EXT}'
    current_description_path = DRIVE_PATH / f'source/{DESCRIPTION_BASE_NAME}{LANG_SUFFIX}{IMG_EXT}'
    current_architecture_path = DRIVE_PATH / f'source/{ARCHITECTURE_BASE_NAME}{LANG_SUFFIX}{IMG_EXT}'

    with st.expander(t('info_expander_project')):
        if current_presentation_path.exists():
            try: st.image(Image.open(current_presentation_path), use_container_width=True)
            except Exception as e: st.error(f"{t('error_presentation_load')} {current_presentation_path}. Error: {e}")
        else: st.error(f"{t('error_presentation_load')} {current_presentation_path}")

    with st.expander(t('info_expander_description')):
        if current_description_path.exists():
            try: st.image(Image.open(current_description_path), use_container_width=True)
            except Exception as e: st.error(f"{t('error_description_load')} {current_description_path}. Error: {e}")
        else: st.error(f"{t('error_description_load')} {current_description_path}")

    with st.expander(t('info_expander_architecture')):
        if current_architecture_path.exists():
            try: st.image(Image.open(current_architecture_path), use_container_width=True)
            except Exception as e: st.error(f"{t('error_architecture_load')} {current_architecture_path}. Error: {e}")
        else: st.error(f"{t('error_architecture_load')} {current_architecture_path}")
