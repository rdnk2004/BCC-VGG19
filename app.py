import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # silence oneDNN info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # hide TF info & warnings
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"            # hide absl warnings

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import random
import glob

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BloodLens · AI Blood Cell Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — premium dark medical aesthetic
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --border-subtle: rgba(99, 179, 237, 0.12);
    --border-glow: rgba(99, 179, 237, 0.3);
    --accent-blue: #63b3ed;
    --accent-cyan: #22d3ee;
    --accent-purple: #a78bfa;
    --accent-emerald: #34d399;
    --accent-rose: #fb7185;
    --accent-amber: #fbbf24;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gradient-hero: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    --gradient-accent: linear-gradient(135deg, #63b3ed 0%, #a78bfa 50%, #22d3ee 100%);
    --shadow-glow: 0 0 30px rgba(99, 179, 237, 0.15);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header {visibility: hidden;}

/* ── Remove empty Streamlit gaps ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:empty,
[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
.block-container {
    padding-top: 2rem !important;
}

/* ── Fix Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stPlotlyChart"] > div > div > div > .modebar-container {
    display: none !important;
}
.js-plotly-plot .plotly .modebar {
    display: none !important;
}
/* Remove dark tab/header bars above charts */
[data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
}
[data-testid="stTabs"] {
    display: none !important;
}

/* ── Hero Header ── */
.hero-header {
    background: var(--gradient-hero);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-glow);
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99, 179, 237, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(167, 139, 250, 0.04) 0%, transparent 50%);
    animation: pulse-bg 8s ease-in-out infinite;
}
@keyframes pulse-bg {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.05); }
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem 0;
    position: relative;
    letter-spacing: -1px;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    margin: 0;
    position: relative;
    font-weight: 400;
    line-height: 1.7;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent-emerald);
    position: relative;
}
.hero-badge .badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-emerald);
    animation: badge-pulse 2s ease-in-out infinite;
}
@keyframes badge-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(52,211,153,0); }
}

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.glass-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-accent);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.glass-card:hover::after {
    opacity: 1;
}

/* ── Upload Zone ── */
.upload-zone {
    background: linear-gradient(135deg, rgba(99, 179, 237, 0.05) 0%, rgba(167, 139, 250, 0.05) 100%);
    border: 2px dashed var(--border-glow);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    margin: 0.5rem 0;
}
.upload-zone:hover {
    border-color: var(--accent-blue);
    background: linear-gradient(135deg, rgba(99, 179, 237, 0.1) 0%, rgba(167, 139, 250, 0.1) 100%);
}
.upload-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    display: block;
}
.upload-text {
    font-size: 1.05rem;
    color: var(--text-secondary);
}
.upload-text strong {
    color: var(--accent-blue);
}

/* ── Prediction Result ── */
.result-card {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.08) 0%, rgba(99, 179, 237, 0.08) 100%);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeInUp 0.6s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-emerald), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0 0.25rem 0;
}
.result-confidence {
    font-size: 1.15rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Cell Info Card ── */
.cell-info {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1.5rem;
    animation: fadeInUp 0.8s ease;
}
.cell-info h3 {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--accent-blue);
    margin: 0 0 1rem 0;
}
.cell-info p {
    color: var(--text-secondary);
    line-height: 1.7;
    font-size: 0.95rem;
    margin: 0;
}
.cell-info .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.info-item {
    background: rgba(99, 179, 237, 0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
.info-item .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    font-weight: 600;
}
.info-item .value {
    font-size: 1rem;
    color: var(--text-primary);
    font-weight: 600;
    margin-top: 0.2rem;
}

/* ── Model Stats Grid ── */
.model-stats {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.model-stats::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-accent);
}
.model-stats-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    font-weight: 700;
    margin: 0 0 1.2rem 0;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
}
.stat-item {
    text-align: center;
    padding: 0.8rem 0.5rem;
    background: rgba(99, 179, 237, 0.04);
    border-radius: 12px;
    border: 1px solid transparent;
    transition: all 0.3s ease;
}
.stat-item:hover {
    border-color: var(--border-glow);
    background: rgba(99, 179, 237, 0.08);
    transform: translateY(-2px);
}
.stat-item .stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.stat-item.accent-green .stat-value {
    background: linear-gradient(135deg, var(--accent-emerald), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-item .stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-top: 0.3rem;
    font-weight: 600;
}
.stat-item .stat-sub {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
    font-weight: 400;
}

/* ── Sidebar custom ── */
.sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
}
.sidebar-brand .brand-icon {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 0.4rem;
}
.sidebar-brand .brand-name {
    font-size: 1.3rem;
    font-weight: 800;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-section {
    background: rgba(99, 179, 237, 0.05);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.sidebar-section h4 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--accent-blue) !important;
    margin: 0 0 0.6rem 0;
    font-weight: 700;
}
.sidebar-section p {
    font-size: 0.88rem;
    color: var(--text-secondary) !important;
    line-height: 1.6;
    margin: 0.3rem 0;
}

/* ── Streamlit overrides ── */
.stFileUploader > div {
    border: none !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] {
    background: transparent !important;
}
.stButton > button {
    background: var(--gradient-accent) !important;
    color: #0a0e17 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 179, 237, 0.3) !important;
}

/* ── Plotly chart background ── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── Image styling ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Model & class info
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

CELL_INFO = {
    "Eosinophil": {
        "emoji": "🟠",
        "description": "Eosinophils are a type of white blood cell that play a crucial role in fighting parasitic infections and are involved in allergic reactions and inflammation.",
        "normal_range": "1 – 4% of WBCs",
        "lifespan": "8 – 12 days",
        "size": "12 – 17 μm",
        "key_function": "Anti-parasitic defense & allergy response",
        "color_var": "--accent-amber",
    },
    "Lymphocyte": {
        "emoji": "🔵",
        "description": "Lymphocytes are vital for the adaptive immune system. They include T-cells, B-cells, and Natural Killer cells, responsible for targeted immune responses and antibody production.",
        "normal_range": "20 – 40% of WBCs",
        "lifespan": "Years (memory cells)",
        "size": "7 – 15 μm",
        "key_function": "Adaptive immunity & antibody production",
        "color_var": "--accent-blue",
    },
    "Monocyte": {
        "emoji": "🟣",
        "description": "Monocytes are the largest type of white blood cell. They migrate into tissues and differentiate into macrophages or dendritic cells, engulfing pathogens and debris through phagocytosis.",
        "normal_range": "2 – 8% of WBCs",
        "lifespan": "1 – 3 days in blood",
        "size": "15 – 30 μm",
        "key_function": "Phagocytosis & antigen presentation",
        "color_var": "--accent-purple",
    },
    "Neutrophil": {
        "emoji": "🟢",
        "description": "Neutrophils are the most abundant white blood cell. They are first-responder cells at sites of infection, attacking bacteria through phagocytosis, degranulation, and neutrophil extracellular traps.",
        "normal_range": "40 – 70% of WBCs",
        "lifespan": "5 – 90 hours",
        "size": "12 – 15 μm",
        "key_function": "First-line bacterial defense",
        "color_var": "--accent-emerald",
    },
}

ACCENT_COLORS = {
    "Eosinophil": "#fbbf24",
    "Lymphocyte": "#63b3ed",
    "Monocyte": "#a78bfa",
    "Neutrophil": "#34d399",
}


# ──────────────────────────────────────────────────────────────
# Load model  (cached so it loads only once)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    model_path = os.path.join(os.path.dirname(__file__), "VGG19_finetuned_best.h5")
    model = tf.keras.models.load_model(model_path)
    return model


# ──────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.vgg19 import preprocess_input

    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


# ──────────────────────────────────────────────────────────────
# Plotly charts
# ──────────────────────────────────────────────────────────────
def create_donut_chart(predicted_class: str, confidence: float):
    colors = [ACCENT_COLORS.get(predicted_class, "#63b3ed"), "rgba(30,41,59,0.5)"]
    fig = go.Figure(
        go.Pie(
            values=[confidence, 100 - confidence],
            hole=0.75,
            marker=dict(colors=colors, line=dict(width=0)),
            textinfo="none",
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        text=f"<b>{confidence:.1f}%</b>",
        font=dict(size=28, color="#f1f5f9", family="Inter"),
        showarrow=False,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_bar_chart(class_names: list, probabilities: list):
    colors = [ACCENT_COLORS.get(c, "#63b3ed") for c in class_names]
    sorted_indices = np.argsort(probabilities)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_probs = [probabilities[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    fig = go.Figure(
        go.Bar(
            x=sorted_probs,
            y=sorted_names,
            orientation="h",
            marker=dict(
                color=sorted_colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{p:.1f}%" for p in sorted_probs],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=13, family="Inter"),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(
            range=[0, max(sorted_probs) * 1.25],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#94a3b8", size=13, family="Inter"),
        ),
        margin=dict(t=10, b=10, l=10, r=50),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">🔬</span>
        <span class="brand-name">BloodLens</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>📋 How to Use</h4>
        <p>1. Upload a blood cell microscopy image (JPG or PNG).</p>
        <p>2. The AI model will instantly classify the cell type.</p>
        <p>3. Review confidence scores and cell information.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>🧠 Model Details</h4>
        <p><strong>Architecture:</strong> VGG-19 (fine-tuned)</p>
        <p><strong>Input Size:</strong> 224 × 224 px (RGB)</p>
        <p><strong>Classes:</strong> 4 WBC types</p>
        <p><strong>Framework:</strong> TensorFlow / Keras</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>🩸 Supported Cell Types</h4>
        <p>🟠 Eosinophil</p>
        <p>🔵 Lymphocyte</p>
        <p>🟣 Monocyte</p>
        <p>🟢 Neutrophil</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>⚠️ Disclaimer</h4>
        <p style="font-size:0.82rem;">This tool is for <strong>educational and research purposes only</strong>. 
        It is not intended for clinical diagnosis. Always consult a qualified medical professional.</p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero-header">
    <p class="hero-title">BloodLens — AI Blood Cell Classifier</p>
    <p class="hero-subtitle">Upload a blood cell microscopy image and let our deep-learning model identify the cell type instantly. Powered by a fine-tuned VGG-19 convolutional neural network.</p>
    <div class="hero-badge">
        <span class="badge-dot"></span>
        Model Accuracy: 88.14%
    </div>
</div>
""", unsafe_allow_html=True)

# Model Statistics
st.markdown("""
<div class="model-stats">
    <div class="model-stats-title">📊 Model Performance & Architecture</div>
    <div class="stats-grid">
        <div class="stat-item accent-green">
            <div class="stat-value">88.14%</div>
            <div class="stat-label">Accuracy</div>
            <div class="stat-sub">Test Set</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">VGG-19</div>
            <div class="stat-label">Architecture</div>
            <div class="stat-sub">Fine-tuned</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">4</div>
            <div class="stat-label">Cell Classes</div>
            <div class="stat-sub">WBC Types</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">224²</div>
            <div class="stat-label">Input Size</div>
            <div class="stat-sub">RGB Pixels</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">~20M</div>
            <div class="stat-label">Parameters</div>
            <div class="stat-sub">Trainable</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Upload Section
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="padding:0.8rem 1.2rem;">
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
        <div style="width:30px; height:30px; border-radius:8px; background:rgba(99,179,237,0.12); display:flex; align-items:center; justify-content:center; font-size:0.95rem;">📤</div>
        <span style="font-size:1.05rem; font-weight:700; color:var(--text-primary);">Upload Blood Cell Images</span>
        <span style="font-size:0.75rem; color:var(--text-muted); margin-left:auto;">Multiple files supported</span>
    </div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose microscopy images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("""
    <div class="upload-zone">
        <span class="upload-icon">🩸</span>
        <p class="upload-text">Drag & drop or <strong>browse</strong> blood cell images<br>
        <small style="color: var(--text-muted);">Supports JPG and PNG • Upload multiple files at once</small></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Quick Test Section — random sample images
# ──────────────────────────────────────────────────────────────
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TEST_SIMPLE")
has_test_images = os.path.isdir(TEST_DIR)

use_samples = False
sample_images = []

if has_test_images and not uploaded_files:
    # Gather all test images
    all_test_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_test_images.extend(glob.glob(os.path.join(TEST_DIR, "**", ext), recursive=True))

    if all_test_images:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.5rem; margin:1.5rem 0 0.8rem 0;">
            <div style="flex:1; height:1px; background:var(--border-subtle);"></div>
            <span style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; font-weight:600;">or try sample images</span>
            <div style="flex:1; height:1px; background:var(--border-subtle);"></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎲 Pick 5 Random Test Images", use_container_width=True, type="primary"):
            st.session_state["sample_seed"] = random.randint(0, 999999)

        if "sample_seed" in st.session_state:
            rng = random.Random(st.session_state["sample_seed"])
            count = min(5, len(all_test_images))
            sample_paths = rng.sample(all_test_images, count)
            sample_images = []
            for p in sample_paths:
                img = Image.open(p)
                # Get the parent folder name as the true label
                true_label = os.path.basename(os.path.dirname(p))
                sample_images.append((img, os.path.basename(p), true_label))
            use_samples = True


# ──────────────────────────────────────────────────────────────
# Determine which images to process
# ──────────────────────────────────────────────────────────────
images_to_process = []  # list of (PIL.Image, filename, true_label_or_None)

if uploaded_files:
    for f in uploaded_files:
        images_to_process.append((Image.open(f), f.name, None))
elif use_samples:
    images_to_process = sample_images


# ──────────────────────────────────────────────────────────────
# Results Section — one card per image
# ──────────────────────────────────────────────────────────────
if images_to_process:
    # Load model once
    with st.spinner("🧠 Loading model…"):
        model = load_model()

    # Summary counter
    summary = {}

    st.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:0.6rem;
        margin: 2rem 0 1rem 0;
    ">
        <div style="width:36px; height:36px; border-radius:10px; background:rgba(52,211,153,0.12); display:flex; align-items:center; justify-content:center; font-size:1.1rem;">🔍</div>
        <span style="font-size:1.15rem; font-weight:700; color:var(--text-primary);">Analysis Results</span>
        <span style="
            background: rgba(99,179,237,0.12);
            border: 1px solid rgba(99,179,237,0.25);
            border-radius: 50px;
            padding: 0.2rem 0.8rem;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent-blue);
            margin-left: auto;
        ">{len(images_to_process)} image{'s' if len(images_to_process) > 1 else ''}{' • Sample' if use_samples else ''}</span>
    </div>
    """, unsafe_allow_html=True)

    for idx, (image, filename, true_label) in enumerate(images_to_process):


        # Preprocess & predict
        preprocessed = preprocess_image(image)
        preds = model.predict(preprocessed, verbose=0)
        probs = preds[0]

        # Handle softmax vs sigmoid output
        if probs.sum() < 0.99 or probs.sum() > 1.01:
            exp_preds = np.exp(probs - np.max(probs))
            probs = exp_preds / exp_preds.sum()

        # Dynamically adjust class names
        num_classes = len(probs)
        if num_classes != len(CLASS_NAMES):
            class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            class_names = CLASS_NAMES

        predicted_idx = int(np.argmax(probs))
        predicted_class = class_names[predicted_idx]
        confidence = float(probs[predicted_idx]) * 100
        all_probs = [float(p) * 100 for p in probs]

        # Track for summary
        summary[predicted_class] = summary.get(predicted_class, 0) + 1

        # ── Divider between results ──
        if idx > 0:
            st.markdown('<hr style="border:none; border-top:1px solid var(--border-subtle); margin:1.5rem 0;">', unsafe_allow_html=True)

        # ── Per-image result ──
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown(f"""
            <div class="glass-card" style="padding:1rem;">
                <div style="font-size:0.8rem; color:var(--text-muted); font-weight:600; margin-bottom:0.5rem;">
                    IMAGE {idx + 1} — {filename}{f' <span style="color:#34d399; font-weight:700;">({true_label})</span>' if true_label else ''}
                </div>
            """, unsafe_allow_html=True)
            st.image(image, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_res:
            # Result card
            emoji = CELL_INFO.get(predicted_class, {}).get("emoji", "🔬")
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:3rem;">{emoji}</div>
                <div class="result-label">{predicted_class}</div>
                <div class="result-confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Charts
            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                st.markdown('<div class="glass-card" style="padding:1rem;">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_donut_chart(predicted_class, confidence),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"donut_{idx}",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with chart_c2:
                st.markdown('<div class="glass-card" style="padding:1rem;">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_bar_chart(class_names, all_probs),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"bar_{idx}",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Cell information
            if predicted_class in CELL_INFO:
                info = CELL_INFO[predicted_class]
                st.markdown(f"""
                <div class="cell-info">
                    <h3>{info["emoji"]} About {predicted_class}s</h3>
                    <p>{info["description"]}</p>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="label">Normal Range</div>
                            <div class="value">{info["normal_range"]}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Lifespan</div>
                            <div class="value">{info["lifespan"]}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Cell Size</div>
                            <div class="value">{info["size"]}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Key Function</div>
                            <div class="value">{info["key_function"]}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Summary bar (when multiple images) ──
    if len(images_to_process) > 1:
        def hex_to_rgba(hex_color, alpha):
            """Convert #rrggbb to rgba(r,g,b,alpha)."""
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        summary_pills = ""
        for cell_type, count in sorted(summary.items(), key=lambda x: -x[1]):
            emoji = CELL_INFO.get(cell_type, {}).get("emoji", "🔬")
            color = ACCENT_COLORS.get(cell_type, "#63b3ed")
            bg = hex_to_rgba(color, 0.07)
            border = hex_to_rgba(color, 0.25)
            summary_pills += f'<div style="background:{bg}; border:1px solid {border}; border-radius:12px; padding:0.8rem 1.2rem; text-align:center; flex:1; min-width:120px;">'
            summary_pills += f'<div style="font-size:1.5rem;">{emoji}</div>'
            summary_pills += f'<div style="font-size:1.3rem; font-weight:800; color:{color};">{count}</div>'
            summary_pills += f'<div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; font-weight:600;">{cell_type}</div>'
            summary_pills += '</div>'

        st.markdown(f"""
        <div class="glass-card" style="margin-top:2rem;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;">
                <div style="width:36px; height:36px; border-radius:10px; background:rgba(167,139,250,0.12); display:flex; align-items:center; justify-content:center; font-size:1.1rem;">📊</div>
                <span style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">Batch Summary — {len(images_to_process)} Images Analyzed</span>
            </div>
            <div style="display:flex; gap:1rem; flex-wrap:wrap;">
                {summary_pills}
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:4rem 2rem; margin-top:1rem;">
        <div style="font-size:4rem; margin-bottom:1rem; opacity:0.5;">🔬</div>
        <p style="font-size:1.2rem; color:var(--text-secondary); margin:0;">
            Upload images to see the <strong style="color:var(--accent-blue);">AI prediction</strong>
        </p>
        <p style="font-size:0.9rem; color:var(--text-muted); margin-top:0.5rem;">
            Results will appear here with confidence charts and cell information
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="
    text-align:center;
    padding:2rem 0 1rem 0;
    color:var(--text-muted);
    font-size:0.82rem;
    border-top:1px solid var(--border-subtle);
    margin-top:3rem;
">
    Built with ❤️ using <strong>Streamlit</strong> &amp; <strong>TensorFlow</strong> · 
    VGG-19 Fine-tuned Model · For Educational Purposes Only
</div>
""", unsafe_allow_html=True)
