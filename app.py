"""
Facial Emotion Recognition — Premium Streamlit Frontend (Gold / Amber theme)
=============================================================================
A polished, SaaS-grade UI wrapped around a pretrained MobileNetV2
Keras model (emotion_recognition.keras).

Note: this file only builds the interface around the existing model.
No training or model-architecture code lives here.
"""

import io
import time
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# TensorFlow is imported lazily inside load_model() so the rest of the
# app can still render (and show a friendly error) if TF is missing.

# ------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Constants / configuration
# ------------------------------------------------------------------
MODEL_PATH = "emotion_recognition.keras"
IMG_SIZE = (160, 160)  # matches the model's input_shape (160, 160, 3)

# Class order follows the conventional alphabetical class_indices that
# Keras' ImageDataGenerator / image_dataset_from_directory assigns by
# default. If your training pipeline used a different folder order,
# just reorder this list to match your saved class_indices mapping.
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_META = {
    "Happy":    {"emoji": "😀", "color": "#FFD54F", "desc": "raised cheeks, a widened mouth, and relaxed brows"},
    "Sad":      {"emoji": "😢", "color": "#7DA6FF", "desc": "downturned lips, drooping eyelids, and inner brows pulled up"},
    "Angry":    {"emoji": "😠", "color": "#FF6B6B", "desc": "lowered, tightened brows and a firmly pressed mouth"},
    "Fear":     {"emoji": "😨", "color": "#C99BFF", "desc": "widened eyes, raised brows, and a tensed, pulled-back mouth"},
    "Neutral":  {"emoji": "😐", "color": "#B0B6C4", "desc": "relaxed, symmetrical features with no strong muscle activation"},
    "Disgust":  {"emoji": "🤢", "color": "#8AE68A", "desc": "a wrinkled nose, raised upper lip, and narrowed eyes"},
    "Surprise": {"emoji": "😲", "color": "#FFB300", "desc": "raised brows, widened eyes, and a dropped-open jaw"},
}

# ------------------------------------------------------------------
# Global CSS — dark gold/amber glassmorphism SaaS theme
# ------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

            #MainMenu, footer {visibility: hidden;}
            div[data-testid="stToolbar"] {visibility: hidden; height: 0;}
            div[data-testid="stDecoration"] {visibility: hidden;}
            header[data-testid="stHeader"] {background: transparent;}

            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

            .stApp {
                background:
                    radial-gradient(circle at 15% 0%, rgba(255, 193, 7, 0.16), transparent 45%),
                    radial-gradient(circle at 85% 15%, rgba(255, 111, 0, 0.12), transparent 45%),
                    #0D0B07;
                color: #F1EDE3;
            }

            .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

            /* ---------- Hero ---------- */
            .hero {
                text-align: center;
                padding: 2.6rem 1.5rem 2.2rem 1.5rem;
                margin-bottom: 1.6rem;
            }
            .hero h1 {
                font-family: 'Sora', sans-serif;
                font-weight: 800;
                font-size: 3.1rem;
                letter-spacing: -0.03em;
                margin: 0;
                background: linear-gradient(100deg, #FFE082 0%, #FFB300 45%, #FF8F00 100%);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }
            .hero p.subtitle {
                font-size: 1.08rem;
                color: #B7AD98;
                margin-top: 0.65rem;
                font-weight: 500;
            }
            .badges { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.6rem; margin-top: 1.4rem; }
            .badge {
                background: rgba(255,193,7,0.07);
                border: 1px solid rgba(255,193,7,0.18);
                backdrop-filter: blur(6px);
                padding: 0.45rem 1rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 600;
                color: #F0DFAE;
            }

            /* ---------- Glass card (real container, not a split div) ---------- */
            div[data-testid="stVerticalBlockBorderWrapper"]:has(div.card-anchor) {
                background: rgba(255,193,7,0.045);
                border: 1px solid rgba(255,193,7,0.14);
                border-radius: 20px;
                padding: 0.4rem 0.5rem;
                backdrop-filter: blur(14px);
                box-shadow: 0 8px 30px rgba(0,0,0,0.35);
                margin-bottom: 1.3rem;
            }
            .card-anchor { display: none; }
            .glass h3 {
                font-family: 'Sora', sans-serif;
                font-size: 1.05rem;
                font-weight: 700;
                margin: 0 0 1rem 0;
                color: #FBEFD2;
            }
            .section-eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                font-weight: 700;
                color: #B7AD98;
                margin-bottom: 0.3rem;
            }

            /* ---------- Prediction hero card ---------- */
            .predict-card {
                text-align: center;
                padding: 1.9rem 1.4rem 1.6rem 1.4rem;
                border-radius: 22px;
                background: linear-gradient(160deg, rgba(255,179,0,0.18), rgba(255,111,0,0.08));
                border: 1px solid rgba(255,193,7,0.20);
                margin-bottom: 1.3rem;
            }
            .predict-emoji { font-size: 4.6rem; line-height: 1; }
            .predict-label {
                font-family: 'Sora', sans-serif;
                font-size: 1.9rem;
                font-weight: 800;
                margin-top: 0.3rem;
                color: #FFF4DA;
            }
            .predict-confidence { font-size: 1rem; color: #E3D2A3; margin-top: 0.2rem; font-weight: 600; }

            /* ---------- Progress bars ---------- */
            .bar-row { margin-bottom: 0.85rem; }
            .bar-row:last-child { margin-bottom: 0; }
            .bar-top { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem; }
            .bar-name { font-weight: 600; color: #E8DFC8; }
            .bar-name.top { color: #FFE082; }
            .bar-pct { font-weight: 700; color: #D8CBA6; }
            .bar-track {
                width: 100%; height: 9px; border-radius: 999px;
                background: rgba(255,255,255,0.07); overflow: hidden;
            }
            .bar-fill { height: 100%; border-radius: 999px; }

            /* ---------- Info grid ---------- */
            .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
            .info-cell {
                background: rgba(255,193,7,0.035);
                border: 1px solid rgba(255,193,7,0.10);
                border-radius: 12px;
                padding: 0.65rem 0.85rem;
            }
            .info-cell .k { font-size: 0.72rem; color: #B7AD98; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
            .info-cell .v { font-size: 0.98rem; color: #FBEFD2; font-weight: 700; margin-top: 0.15rem; }

            .explain-box {
                border-left: 3px solid #FFB300;
                background: rgba(255,179,0,0.09);
                padding: 0.9rem 1.1rem;
                border-radius: 10px;
                font-size: 0.94rem;
                color: #ECE0C4;
                line-height: 1.5;
            }

            .error-box {
                background: rgba(255,107,107,0.10);
                border: 1px solid rgba(255,107,107,0.35);
                border-radius: 16px;
                padding: 1.1rem 1.3rem;
                color: #FFD3D3;
                font-weight: 500;
            }
            .warn-box {
                background: rgba(255,209,102,0.10);
                border: 1px solid rgba(255,209,102,0.35);
                border-radius: 16px;
                padding: 1.1rem 1.3rem;
                color: #FFE9B8;
                font-weight: 500;
            }

            [data-testid="stFileUploader"] {
                border-radius: 16px;
                padding: 0.4rem;
                background: rgba(255,193,7,0.03);
                border: 1px dashed rgba(255,193,7,0.22);
            }
            [data-testid="stFileUploader"] section { background: transparent; }

            .footer {
                text-align: center;
                color: #9C927C;
                font-size: 0.88rem;
                margin-top: 2.4rem;
                padding-top: 1.4rem;
                border-top: 1px solid rgba(255,193,7,0.12);
            }
            .footer b { color: #D8CBA6; }

            .stButton>button {
                border-radius: 12px;
                font-weight: 600;
                border: 1px solid rgba(255,193,7,0.18);
                background: linear-gradient(120deg, #FFB300, #FF8F00);
                color: #1A1200;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .stButton>button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 18px rgba(255,179,0,0.35);
            }

            div[data-testid="stProgress"] div[role="progressbar"] > div {
                background: linear-gradient(90deg, #FFD54F, #FF8F00);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def glass_card():
    """Return a Streamlit container styled as a gold-glass card.

    Using st.container(key=...) instead of a raw '<div>' opened in one
    st.markdown() call and closed in another is what makes the styling
    reliably wrap the *real* widgets (file_uploader, image, progress,
    etc.) placed inside it — those widgets become actual DOM children
    of the container, so backgrounds/borders/blur apply correctly and
    nothing renders as an empty box or invisible content.
    """
    container = st.container(border=False, key=f"card_{glass_card.counter}")
    glass_card.counter += 1
    with container:
        st.markdown('<div class="card-anchor"></div>', unsafe_allow_html=True)
    return container


glass_card.counter = 0


# ------------------------------------------------------------------
# Model loading (cached — runs once per process)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load the Keras model once and reuse it across reruns/sessions."""
    import tensorflow as tf  # local import keeps startup fast if TF is slow to import
    model = tf.keras.models.load_model(path)
    return model


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize/convert a PIL image into the (1, 160, 160, 3) float32 array
    the model expects. Normalization/preprocessing is baked into the
    model graph itself, so we only resize and batch here."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def run_prediction(model, pil_image: Image.Image):
    """Run a forward pass and return (probabilities, elapsed_ms)."""
    batch = preprocess_image(pil_image)
    start = time.perf_counter()
    preds = model.predict(batch, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    probs = np.asarray(preds[0], dtype="float64")
    # Guard against a model whose final layer isn't already softmax-normalized
    if not np.isclose(probs.sum(), 1.0, atol=0.02):
        exp = np.exp(probs - probs.max())
        probs = exp / exp.sum()
    return probs, elapsed_ms


def human_file_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def build_explanation(label: str, confidence: float) -> str:
    meta = EMOTION_META[label]
    if confidence >= 0.75:
        certainty = "a strong, confident match"
    elif confidence >= 0.5:
        certainty = "a reasonably confident match"
    else:
        certainty = "the closest match among a fairly even spread"
    return (
        f"The facial features most closely resemble the <b>{label}</b> class based on the "
        f"model's learned representation — specifically {meta['desc']}. "
        f"This was {certainty}, with a confidence score of {confidence * 100:.1f}%."
    )


# ------------------------------------------------------------------
# UI sections
# ------------------------------------------------------------------
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Facial Emotion Recognition</h1>
            <p class="subtitle">AI-powered Facial Expression Analysis using Deep Learning</p>
            <div class="badges">
                <div class="badge">🧠 TensorFlow</div>
                <div class="badge">⚡ MobileNetV2</div>
                <div class="badge">🎯 Multi-Class Classification</div>
                <div class="badge">📸 Image Upload</div>
                <div class="badge">📊 Confidence Score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_column():
    with glass_card():
        st.markdown("<h3>📸 Upload a Face Image</h3>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag and drop a file here, or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        pil_image = None
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            try:
                pil_image = Image.open(io.BytesIO(raw_bytes))
                pil_image.load()
            except (UnidentifiedImageError, OSError):
                st.markdown(
                    '<div class="warn-box">⚠️ That file couldn\'t be read as an image. '
                    "Please upload a valid JPG or PNG.</div>",
                    unsafe_allow_html=True,
                )
                return None

            st.image(pil_image, caption="Preview", width="stretch")

            st.markdown('<div class="section-eyebrow" style="margin-top:1rem;">Image Information</div>', unsafe_allow_html=True)
            w, h = pil_image.size
            st.markdown(
                f"""
                <div class="info-grid">
                    <div class="info-cell"><div class="k">Width</div><div class="v">{w}px</div></div>
                    <div class="info-cell"><div class="k">Height</div><div class="v">{h}px</div></div>
                    <div class="info-cell"><div class="k">File Size</div><div class="v">{human_file_size(len(raw_bytes))}</div></div>
                    <div class="info-cell"><div class="k">Format</div><div class="v">{(pil_image.format or uploaded.type or "Unknown")}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color:#B7AD98; font-size:0.9rem; margin-top:0.6rem;">'
                "Supported formats: JPG, JPEG, PNG</p>",
                unsafe_allow_html=True,
            )

    return pil_image


def render_prediction_column(model, pil_image) -> None:
    with glass_card():
        st.markdown("<h3>🔮 Prediction</h3>", unsafe_allow_html=True)

        if pil_image is None:
            st.markdown(
                '<p style="color:#B7AD98; font-size:0.92rem;">Upload an image on the left to '
                "see the predicted emotion here.</p>",
                unsafe_allow_html=True,
            )
            return

        if model is None:
            return

        with st.spinner("Analyzing facial features..."):
            try:
                probs, elapsed_ms = run_prediction(model, pil_image)
            except Exception as exc:  # noqa: BLE001 — surface any inference issue gracefully
                st.markdown(
                    f'<div class="error-box">🚫 <b>Prediction failed.</b><br>'
                    f"The model could not process this image. Details: {type(exc).__name__}.</div>",
                    unsafe_allow_html=True,
                )
                return

        top_idx = int(np.argmax(probs))
        top_label = EMOTION_CLASSES[top_idx]
        top_conf = float(probs[top_idx])
        meta = EMOTION_META[top_label]

        st.markdown(
            f"""
            <div class="predict-card">
                <div class="predict-emoji">{meta['emoji']}</div>
                <div class="predict-label">{top_label}</div>
                <div class="predict-confidence">Confidence: {top_conf * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(top_conf, 0.0), 1.0))

        st.markdown(
            f"""
            <div class="info-grid" style="margin-top:0.7rem;">
                <div class="info-cell"><div class="k">Prediction Time</div><div class="v">{elapsed_ms:.0f} ms</div></div>
                <div class="info-cell"><div class="k">Analyzed At</div><div class="v">{datetime.now().strftime('%H:%M:%S')}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-eyebrow" style="margin-top:1.2rem;">Why this result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="explain-box">{build_explanation(top_label, top_conf)}</div>',
            unsafe_allow_html=True,
        )

    render_probability_card(probs, top_label)


def render_probability_card(probs: np.ndarray, top_label: str) -> None:
    with glass_card():
        st.markdown("<h3>📊 Emotion Probability Breakdown</h3>", unsafe_allow_html=True)

        order = np.argsort(probs)[::-1]
        rows_html = ""
        for idx in order:
            label = EMOTION_CLASSES[idx]
            pct = float(probs[idx]) * 100
            meta = EMOTION_META[label]
            is_top = label == top_label
            name_class = "bar-name top" if is_top else "bar-name"
            rows_html += f"""
            <div class="bar-row">
                <div class="bar-top">
                    <span class="{name_class}">{meta['emoji']} {label}{' — Predicted' if is_top else ''}</span>
                    <span class="bar-pct">{pct:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%; background:{meta['color']};"></div>
                </div>
            </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)


def render_model_info() -> None:
    with glass_card():
        st.markdown("<h3>⚙️ Model Information</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-grid">
                <div class="info-cell"><div class="k">Model Name</div><div class="v">MobileNetV2</div></div>
                <div class="info-cell"><div class="k">Input Size</div><div class="v">160 × 160</div></div>
                <div class="info-cell"><div class="k">Classes</div><div class="v">7</div></div>
                <div class="info-cell"><div class="k">Framework</div><div class="v">TensorFlow</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            Made with ❤️ using <b>TensorFlow</b> · <b>Streamlit</b> · <b>Python</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# App entry point
# ------------------------------------------------------------------
def main() -> None:
    inject_css()
    render_hero()

    model = None
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 — show a friendly error instead of crashing
        st.markdown(
            f"""
            <div class="error-box">
                🚫 <b>Model failed to load.</b><br>
                Make sure <code>{MODEL_PATH}</code> is in the same folder as this app.<br>
                <span style="font-size:0.85rem; opacity:0.8;">Details: {type(exc).__name__}: {exc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        pil_image = render_upload_column()
    with right:
        render_prediction_column(model, pil_image)

    render_model_info()
    render_footer()


if __name__ == "__main__":
    main()
