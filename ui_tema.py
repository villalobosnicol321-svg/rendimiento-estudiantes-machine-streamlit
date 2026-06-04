"""Tema visual, traducciones y estilos compartidos entre apps."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Paleta: académico moderno (violeta + coral + turquesa) ---
COLOR_PRIMARIO = "#5B4B8A"
COLOR_SECUNDARIO = "#FF6B6B"
COLOR_ACENTO = "#4ECDC4"
COLOR_AZUL = "#2563EB"
COLOR_AZUL_CLARO = "#3B82F6"
COLOR_AZUL_OSCURO = "#1D4ED8"
COLOR_FONDO = "#F7F5FB"
COLOR_TEXTO = "#2D3142"
COLOR_SUPERFICIE = "#1E293B"
COLOR_SUPERFICIE_CLARA = "#334155"

COLORES_NIVEL = {
    0: "#C1121F",
    1: "#E85D04",
    2: "#2A9D8F",
}
COLORES_NIVEL_NOMBRE = {
    "Deficiente": "#C1121F",
    "Básico": "#E85D04",
    "Superior": "#2A9D8F",
}

ETIQUETAS_VARIABLES = {
    "Hours_Studied": "Horas de estudio semanal",
    "Attendance": "Asistencia (%)",
    "Parental_Involvement": "Participación de los padres",
    "Access_to_Resources": "Acceso a recursos",
    "Extracurricular_Activities": "Actividades extracurriculares",
    "Sleep_Hours": "Horas de sueño",
    "Previous_Scores": "Notas anteriores",
    "Motivation_Level": "Nivel de motivación",
    "Internet_Access": "Acceso a internet",
    "Tutoring_Sessions": "Sesiones de tutoría",
    "Family_Income": "Ingreso familiar",
    "Teacher_Quality": "Calidad del docente",
    "School_Type": "Tipo de colegio",
    "Peer_Influence": "Influencia de pares",
    "Physical_Activity": "Actividad física (horas)",
    "Learning_Disabilities": "Dificultades de aprendizaje",
    "Parental_Education_Level": "Educación de los padres",
    "Distance_from_Home": "Distancia al colegio",
    "Gender": "Género",
    "Exam_Score": "Nota del examen",
    "Nivel": "Nivel de desempeño",
}

TRADUCIR_VALOR = {
    "Low": "Bajo",
    "Medium": "Medio",
    "High": "Alto",
    "Yes": "Sí",
    "No": "No",
    "Male": "Masculino",
    "Female": "Femenino",
    "Public": "Público",
    "Private": "Privado",
    "Near": "Cerca",
    "Moderate": "Moderada",
    "Far": "Lejana",
    "Positive": "Positiva",
    "Negative": "Negativa",
    "Neutral": "Neutra",
    "High School": "Bachillerato",
    "College": "Universidad",
    "Postgraduate": "Posgrado",
}

INVERTIR_VALOR = {v: k for k, v in TRADUCIR_VALOR.items()}

NOMBRES_MODELO_ES = {
    "Gradient Boosting": "Potenciación de gradiente",
    "Regresión Logística": "Regresión logística",
    "RL + SMOTE": "Regresión logística + SMOTE",
    "Random Forest": "Bosque aleatorio",
    "Árbol de Decisión": "Árbol de decisión",
    "PCA + K-Means": "PCA + K-Means",
}

CSS_APP = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .bloque-hero {
        background: linear-gradient(135deg, #5B4B8A 0%, #7B68B8 45%, #4ECDC4 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(91, 75, 138, 0.25);
    }
    .bloque-hero h1 {
        color: white !important;
        font-size: 1.85rem !important;
        margin: 0 0 0.35rem 0 !important;
    }
    .bloque-hero p {
        color: rgba(255,255,255,0.92) !important;
        margin: 0;
        font-size: 1rem;
    }
    .tarjeta-metrica {
        background: white;
        border: 1px solid #E8E4F0;
        border-left: 4px solid #5B4B8A;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 8px rgba(45, 49, 66, 0.06);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span {
        color: #E2E8F0 !important;
    }
    /* Filtros: etiquetas multiselect en azul */
    div[data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #2563EB !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebar"] [data-baseweb="tag"] svg {
        fill: #FFFFFF !important;
    }
    /* Sliders azules en barra lateral */
    div[data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
        background: #2563EB !important;
    }
    div[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
        background: #3B82F6 !important;
        border: 2px solid #DBEAFE !important;
    }
    div[data-testid="stSidebar"] .stSlider [data-baseweb="thumb"] {
        background-color: #3B82F6 !important;
        border-color: #DBEAFE !important;
    }
    /* Pestañas legibles (activa e inactiva) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #334155 !important;
        color: #E2E8F0 !important;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        border: 1px solid #475569 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #475569 !important;
        color: #FFFFFF !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
    }
    .stTabs [aria-selected="false"] {
        background-color: #334155 !important;
        color: #E2E8F0 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: rgba(30, 41, 59, 0.35);
        border-radius: 0 12px 12px 12px;
        padding: 1rem 0.25rem;
        border: 1px solid #475569;
    }
    /* Navegación por radio (secciones) */
    div[data-testid="stRadio"] > div {
        gap: 0.5rem;
    }
    div[data-testid="stRadio"] label {
        background-color: #334155 !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        padding: 0.55rem 1rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p {
        color: #E2E8F0 !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
        border-color: #60A5FA !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #475569 !important;
        border-color: #60A5FA !important;
    }
    .bloque-seccion {
        background: #1E293B;
        border: 1px solid #475569;
        border-left: 4px solid #2563EB;
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
    }
    .bloque-seccion h3 {
        color: #93C5FD !important;
        margin: 0 0 0.5rem 0 !important;
        font-size: 1.15rem !important;
    }
    .bloque-seccion p {
        color: #CBD5E1 !important;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
        margin: 1rem 0 1.25rem 0;
    }
    .kpi-card {
        background: linear-gradient(145deg, #1E293B 0%, #334155 100%);
        border: 1px solid #475569;
        border-top: 3px solid #2563EB;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .kpi-card .kpi-label {
        color: #94A3B8;
        font-size: 0.78rem;
        margin-bottom: 0.25rem;
    }
    .kpi-card .kpi-value {
        color: #F8FAFC;
        font-size: 1.45rem;
        font-weight: 700;
    }
    @media (max-width: 1100px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #5B4B8A, #7B68B8);
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #4A3D72, #6A58A8);
        color: white;
    }
    [data-testid="stMetricValue"] {
        color: #5B4B8A;
        font-weight: 700;
    }
</style>
"""


def aplicar_tema():
    st.markdown(CSS_APP, unsafe_allow_html=True)
    pio.templates["academico"] = go.layout.Template(
        layout=dict(
            font=dict(family="Plus Jakarta Sans, sans-serif", color=COLOR_TEXTO),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(247,245,251,0.8)",
            colorway=[COLOR_PRIMARIO, COLOR_SECUNDARIO, COLOR_ACENTO, "#7B68B8", "#FFD166"],
            title=dict(font=dict(size=16, color=COLOR_PRIMARIO)),
        )
    )
    pio.templates.default = "academico"


def etiqueta_variable(nombre_col: str) -> str:
    return ETIQUETAS_VARIABLES.get(nombre_col, nombre_col.replace("_", " "))


def valor_a_es(valor) -> str:
    s = str(valor)
    return TRADUCIR_VALOR.get(s, s)


def valor_a_en(valor_es: str) -> str:
    return INVERTIR_VALOR.get(valor_es, valor_es)


def traducir_nombre_feature(nombre_tecnico: str) -> str:
    """Traduce nombres de features del pipeline (num__Hours_Studied, cat__Gender_Male...)."""
    texto = nombre_tecnico
    for en, es in sorted(ETIQUETAS_VARIABLES.items(), key=lambda x: -len(x[0])):
        texto = texto.replace(en, es)
    for en, es in TRADUCIR_VALOR.items():
        texto = texto.replace(en, es)
    texto = texto.replace("num__", "").replace("cat__", "").replace("__", " · ")
    return texto


def preparar_datos_es(df: pd.DataFrame) -> pd.DataFrame:
    """Copia del dataset con columnas y categorías en español para visualización."""
    out = df.copy()
    out = out.rename(columns=ETIQUETAS_VARIABLES)
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).map(lambda x: TRADUCIR_VALOR.get(x, x))
    if "Nivel de desempeño" not in out.columns and "Exam_Score" in df.columns:
        out["Nivel de desempeño"] = pd.cut(
            df["Exam_Score"],
            bins=[-1, 64, 74, 100],
            labels=["Deficiente", "Básico", "Superior"],
        )
    elif "Nivel" in df.columns:
        out["Nivel de desempeño"] = df["Nivel"]
    return out


def titulo_seccion(texto: str, descripcion: str = ""):
    desc = f'<p style="color:#94A3B8;margin:0 0 0.5rem 0;font-size:0.9rem;">{descripcion}</p>' if descripcion else ""
    st.markdown(
        f'<div class="bloque-seccion"><h3>{texto}</h3>{desc}',
        unsafe_allow_html=True,
    )


def cerrar_seccion():
    st.markdown("</div>", unsafe_allow_html=True)


def fila_kpis(items: list[tuple[str, str, str]]):
    """items: [(emoji+label, valor, sufijo opcional en label)]"""
    tarjetas = []
    for etiqueta, valor, _ in items:
        tarjetas.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{etiqueta}</div>'
            f'<div class="kpi-value">{valor}</div></div>'
        )
    st.markdown(
        f'<div class="kpi-grid">{"".join(tarjetas)}</div>',
        unsafe_allow_html=True,
    )


def hero(titulo: str, subtitulo: str, icono: str = "📖"):
    st.markdown(
        f"""
        <div class="bloque-hero">
            <h1>{icono} {titulo}</h1>
            <p>{subtitulo}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def estilo_plotly(fig):
    fig.update_layout(
        template="academico",
        margin=dict(l=40, r=24, t=48, b=40),
    )
    return fig
