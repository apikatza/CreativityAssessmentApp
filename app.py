import joblib
import streamlit as st
from PIL import Image
import numpy as np
import torch
import tensorflow as tf
import clip
import fasttext
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras import backend as K
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# —– Monkey‑patch para streamlit-drawable-canvas v0.9.3 —–
import streamlit.elements.lib.image_utils as _image_utils
import streamlit.elements.image as _st_image_mod
_st_image_mod.image_to_url = _image_utils.image_to_url

def specificity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)) * K.round(K.clip(1 - y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip(1 - y_true, 0, 1)) * K.round(K.clip(y_pred, 0, 1)))
    possible_negatives = true_negatives + false_positives
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity


def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Redondear las predicciones a 0 o 1
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # Verdaderos positivos
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  # Total predicho como positivo
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))  # Total de positivos reales

    precision = tp / (predicted_positives + K.epsilon())  # Precisión
    recall = tp / (possible_positives + K.epsilon())  # Recall (sensibilidad)
    
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())  # F1-score
    return f1_val

def specificity_multi(y_true, y_pred):
    # Convertir one-hot a índices de clase
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)

    # Calcular verdaderos negativos (TN) y falsos positivos (FP)
    true_negatives = K.sum(K.cast(K.equal(y_true_class, 0) & K.equal(y_pred_class, 0), 'float32'))
    false_positives = K.sum(K.cast(K.equal(y_true_class, 0) & K.not_equal(y_pred_class, 0), 'float32'))
    
    # Posibles negativos (TN + FP)
    possible_negatives = true_negatives + false_positives
    
    # Especificidad
    specificity_val = true_negatives / (possible_negatives + K.epsilon())
    return specificity_val

def f1_score_multi(y_true, y_pred):
    # Convertir one-hot a índices de clase
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)

    # Verdaderos positivos (TP)
    tp = K.sum(K.cast(K.equal(y_true_class, y_pred_class), 'float32'))

    # Total de predicciones positivas (TP + FP)
    predicted_positives = K.sum(K.cast(K.greater(y_pred_class, 0), 'float32'))

    # Total de verdaderos positivos (TP + FN)
    possible_positives = K.sum(K.cast(K.greater(y_true_class, 0), 'float32'))

    # Calcular precisión y recall
    precision = tp / (predicted_positives + K.epsilon())
    recall = tp / (possible_positives + K.epsilon())

    # Calcular F1-score
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Definir R^2 (Coeficiente de determinación)
def r2_score(y_true, y_pred):
    total = K.sum(K.square(y_true - K.mean(y_true)))
    residual = K.sum(K.square(y_true - y_pred))
    return 1 - residual / (total + K.epsilon())

# —– Funciones de carga y evaluación de tus modelos —–
@st.cache_resource(show_spinner=False)
def load_models():
    with st.spinner("Cargando, por favor espera... Esto puede tardar hasta 15 minutos la primera vez."):
        try:
            with st.spinner("Cargando modelos..."):
                model_keras_FLE = tf.keras.models.load_model("models/model_FLE.keras", custom_objects={'specificity_multi': specificity_multi, "f1_score_multi": f1_score_multi})
                model_keras_T = tf.keras.models.load_model("models/model_T.keras", custom_objects={'specificity': specificity, "f1_score": f1_score})

            with st.spinner("Cargando modelos de lenguaje..."):
                beto_model_name = "dccuchile/bert-base-spanish-wwm-cased"
                beto_tokenizer = AutoTokenizer.from_pretrained(beto_model_name)
                beto_model = TFAutoModel.from_pretrained(beto_model_name)

                fasttext_model_path = 'models/embedding_models/cc.es.300.bin'
                fasttext_model = fasttext.load_model(fasttext_model_path)

            with st.spinner("Cargando modelo CLIP..."):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
                clip_classifier = joblib.load("models/model_O.pkl")

            return model_keras_FLE, model_keras_T, beto_tokenizer, beto_model, fasttext_model, model_clip, preprocess_clip, clip_classifier, device
        except Exception as e:
            st.error(f"Error al cargar modelos: {e}")
            raise

# Crear embedding del título con BETO para un solo texto
def get_beto_embedding(text, tokenizer, model, max_length=128):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=max_length)
    outputs = model(inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()

def evaluate_image_and_title(image: Image.Image, title: str, models):
    model_keras_FLE, model_keras_T, beto_tokenizer, beto_model, fasttext_model, model_clip, preprocess_clip, clip_classifier, device = st.session_state.models

    results = {}

    # --- Preprocesamiento estilo Keras para los 3 primeros modelos ---
    keras_img_size = (224, 224)
    image_keras = image.resize(keras_img_size)
    img_array = img_to_array(image_keras)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

    beto_embedding = get_beto_embedding(title, beto_tokenizer, beto_model)

    fasttext_embedding = fasttext_model.get_sentence_vector(title)           

    # --- Preprocesamiento para CLIP (solo para 'O') ---
    # --- Embeddings CLIP ---
    image_clip = preprocess_clip(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([title]).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(image_clip)
        text_features = model_clip.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Convertir a numpy
    image_emb_np = image_features.cpu().numpy()
    text_emb_np = text_features.cpu().numpy()

    # Concatenar embeddings imagen + texto
    combined_emb = np.concatenate([image_emb_np, text_emb_np], axis=1)

    # Clasificación con los modelos
    pred_proba = clip_classifier.predict_proba(combined_emb)
    pred_label = pred_proba[0][1] * 100  # Probabilidad de que sea "sí" en porcentaje
    results["Originalidad del Dibujo"] = f"{pred_label:.2f}%"

    # Aquí usarías tus modelos Keras reales:
    pred3 = model_keras_T.predict(np.expand_dims(beto_embedding, axis=0))
    pred3 = f"{pred3[0][0] * 100:.2f}%"  # Convertir a porcentaje y añadir símbolo %
    pred2 = model_keras_FLE.predict(np.expand_dims(fasttext_embedding, axis=0))
    top_3_indices = np.argsort(pred2[0])[-3:][::-1]  # Obtener los índices de las 3 clases con mayor probabilidad
    top_3_classes = [(i, f"{pred2[0][i] * 100:.2f}%") for i in top_3_indices]  # Obtener las clases y sus probabilidades en porcentaje

    results["Originalidad del Título"] = pred3
    results["Flexibilidad"] = top_3_classes
    
    return results

# —– Configuración de la página en modo wide —–
st.set_page_config(
    page_title="Evaluador de Dibujo Creativo",
    layout="centered",
    initial_sidebar_state="auto"
)

# Inicializar estado
if 'evaluated' not in st.session_state:
    st.session_state.evaluated = False
    st.session_state.final_img = None
    st.session_state.results = None

# Callback de evaluación
def submit_evaluation():
    base_image = Image.open(st.session_state.base_image_path).convert("RGBA")
    w, h = base_image.size
    canvas_data = st.session_state.get("canvas_data", None)
    title_input = st.session_state.get("title_input", "").strip()

    # Componer imagen
    arr = canvas_data.astype("uint8")
    small_rgba = Image.fromarray(arr, mode="RGBA")
    full_rgba = small_rgba.resize((w, h))
    base_rgba = base_image.convert("RGBA")
    final_rgba = Image.alpha_composite(base_rgba, full_rgba)

    # Añadir marco
    border_size = 1
    frame_color = (0, 0, 0)
    framed_w, framed_h = w + border_size*2, h + border_size*2
    framed_image = Image.new("RGB", (framed_w, framed_h), frame_color)
    framed_image.paste(final_rgba.convert("RGB"), (border_size, border_size))

    # Evaluar modelos
    models = st.session_state.models
    results = evaluate_image_and_title(framed_image, title_input, models)

    # Guardar en session_state
    st.session_state.final_img = framed_image
    st.session_state.results = results
    st.session_state.evaluated = True

# Callback de reinicio
def reset_app():
    st.session_state.evaluated = False
    st.session_state.final_img = None
    st.session_state.results = None
    st.session_state.new_canvas = True

st.title("🎨 Completa y Evalúa tu Dibujo Creativo")

if 'models' not in st.session_state:
    st.session_state.models = load_models()

if st.session_state.evaluated:
    # Dos columnas: Izquierda -> título y dibujo, Derecha -> evaluación
    col1, _, col2 = st.columns([3, 0.5, 3])
    framed = st.session_state.final_img
    w2, h2 = framed.size
    framed_small = framed.resize((w2//2, h2//2))

    with col1:
        st.subheader(f"Título: {st.session_state.title_input}")
        st.image(framed_small, use_container_width=False)

    with col2:
        st.subheader("Evaluación")
        for metric, score in st.session_state.results.items():
            st.write(f"**{metric}:** {score}")
        st.markdown("   ")
        # Botón de reinicio en la primera columna
        col2.button("🔄 Volver a intentar", on_click=reset_app, use_container_width=True)
    
else:
    # Inputs para dibujo y título
    base_drawings_folder = "base_drawings"
    if 'base_image_path' not in st.session_state:
        # Primera carga: inicializamos con una imagen base fija (o puedes usar random aquí también)
        default_image = "base_drawing.png"
        default_path = os.path.join(base_drawings_folder, default_image)
        if not os.path.exists(default_path):
            st.error(f"No se encontró la imagen {default_image} en la carpeta {base_drawings_folder}.")
            st.stop()
        st.session_state.base_image_path = default_path

    # Si se solicitó un nuevo canvas (botón "Volver a intentar")
    if st.session_state.get('new_canvas', False):
        all_drawings = [f for f in os.listdir(base_drawings_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not all_drawings:
            st.error(f"No se encontraron imágenes en la carpeta {base_drawings_folder}.")
            st.stop()
        selected_image = random.choice(all_drawings)
        st.session_state.base_image_path = os.path.join(base_drawings_folder, selected_image)
        st.session_state.new_canvas = False  # Reseteamos para no cambiar otra vez sin botón


    base_image = Image.open(st.session_state.base_image_path).convert("RGBA")
    w, h = base_image.size
    # Mostrar imagen base al 50%
    base_small = base_image.resize((w//2, h//2))

    st.markdown("---")
    st.subheader("1. Completa el dibujo sobre el lienzo")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="#000000",
        background_image=base_small,
        update_streamlit=True,
        height=h//2,
        width=w//2,
        drawing_mode="freedraw",
        key="canvas"
    )
    # Guardar canvas data en estado
    if canvas_result.image_data is not None:
        st.session_state.canvas_data = canvas_result.image_data

    st.markdown("---")
    title_input = st.text_input("2. Ponle un título a tu obra:", key="title_input")

    # --- Validaciones ---
    title_ready = bool(title_input.strip())
    canvas_ready = canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0)


    if not canvas_ready:
        st.error("❌ No se ha detectado ningún trazo nuevo. ¡Dibuja algo primero!")

    if not title_ready:
        st.error("❌ Debes proporcionar un título.")

    # --- Mostrar botón sólo si ambos inputs están listos ---
    if canvas_ready and title_ready:
        st.button("3. Evaluar mi obra", on_click=submit_evaluation)
