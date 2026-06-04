"""
App de predicción multiclase — niveles de desempeño académico.
Proyecto CDP · Sección 13.
"""
from pathlib import Path
 
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
from ui_tema import (
    COLORES_NIVEL,
    NOMBRES_MODELO_ES,
    aplicar_tema,
    etiqueta_variable,
    estilo_plotly,
    hero,
    traducir_nombre_feature,
    valor_a_en,
    valor_a_es,
)
 
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
 
# Nombres en inglés
OPCIONES_MODELO = {
    "Gradient Boosting": "gb",
    "Logistic Regression": "rl",
    "Logistic Regression + SMOTE": "rl_smote",
    "Random Forest": "rf",
    "Decision Tree": "tree",
    "PCA + K-Means": "pca_km",
}
 
ARCHIVOS = {
    "rl": "modelo_regresion_logistica.pkl",
    "rl_smote": "modelo_rl_smote.pkl",
    "tree": "modelo_arbol.pkl",
    "rf": "modelo_random_forest.pkl",
    "gb": "modelo_gradient_boosting.pkl",
    "pca_km": "modelo_pca_kmeans.pkl",
}
 
# Agrupación de campos del formulario
CAMPOS_ACADEMICO_NUM = ["Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions"]
CAMPOS_HABITOS_NUM   = ["Sleep_Hours", "Physical_Activity"]
CAMPOS_HABITOS_CAT   = ["Extracurricular_Activities", "Motivation_Level"]
CAMPOS_SOCIO_CAT     = [
    "Parental_Involvement", "Access_to_Resources", "Internet_Access",
    "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
    "Learning_Disabilities", "Parental_Education_Level", "Distance_from_Home", "Gender",
]
 
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
aplicar_tema()
 
 
@st.cache_resource
def cargar_modelos():
    pipelines = {
        clave: joblib.load(MODELS_DIR / archivo)
        for clave, archivo in ARCHIVOS.items()
    }
    metadata = joblib.load(MODELS_DIR / "metadata.pkl")
    return pipelines, metadata
 
 
if not (MODELS_DIR / "modelo_gradient_boosting.pkl").exists():
    st.error(
        "No se encontraron los modelos entrenados. En la terminal ejecuta: "
        "`python entrenamiento.py` dentro de `rendimiento_estudiantes_app`."
    )
    st.stop()
 
pipelines, metadata = cargar_modelos()
info_modelos = metadata["modelos"]
nombres_nivel = {int(k): v for k, v in metadata["niveles"].items()}
 
 
def construir_entrada(valores_en: dict) -> pd.DataFrame:
    return pd.DataFrame([valores_en])[metadata["columnas_orden"]]
 
 
def alerta_nivel(clase: int):
    nombre = nombres_nivel[clase]
    if clase == 0:
        st.error(f"🔴 **{nombre}** — Se recomienda plan de refuerzo y seguimiento cercano.")
    elif clase == 1:
        st.warning(f"🟠 **{nombre}** — Rendimiento intermedio; hay margen de mejora.")
    else:
        st.success(f"🟢 **{nombre}** — Buen desempeño académico proyectado.")
 
 
def grafico_probabilidades(probs, titulo: str) -> go.Figure:
    etiquetas = [nombres_nivel[i] for i in range(len(probs))]
    colores = [COLORES_NIVEL[i] for i in range(len(probs))]
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=etiquetas,
            orientation="h",
            marker=dict(color=colores, line=dict(color="white", width=1)),
            text=[f"{p * 100:.1f} %" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Estimated probability",
        xaxis_range=[0, 1.08],
        height=300,
        margin=dict(l=20, r=20, t=44, b=20),
    )
    return estilo_plotly(fig)
 
 
def grafico_importancia(pipeline) -> go.Figure | None:
    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    if not hasattr(clf, "feature_importances_"):
        return None
    nombres = [traducir_nombre_feature(n) for n in prep.get_feature_names_out()]
    valores = clf.feature_importances_
    tabla = (
        pd.DataFrame({"variable": nombres, "importancia": valores})
        .sort_values("importancia", ascending=True)
        .tail(12)
    )
    fig = go.Figure(
        go.Bar(
            x=tabla["importancia"],
            y=tabla["variable"],
            orientation="h",
            marker=dict(color="#5B4B8A", line=dict(color="#4ECDC4", width=0.5)),
        )
    )
    fig.update_layout(
        title="Top 12 variables with most influence on the prediction",
        height=440,
        xaxis_title="Relative importance",
    )
    return estilo_plotly(fig)
 
 
def nombre_modelo_en(clave: str) -> str:
    for etiqueta, k in OPCIONES_MODELO.items():
        if k == clave:
            return etiqueta
    return clave
 
 
hero(
    "Student Performance Predictor",
    "Estimates whether the student will reach **Deficient** (≤ 64), **Basic** (65–74) or **Superior** (≥ 75) level. "
    "Includes all 6 models from section 13; recommended: **Gradient Boosting**.",
    "🎯",
)
 
# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Reference model")
    modelo_sel_label = st.selectbox(
        "View detail & interpretation with",
        list(OPCIONES_MODELO.keys()),
        index=0,
    )
    clave_sel = OPCIONES_MODELO[modelo_sel_label]
    info_sel = info_modelos[clave_sel]
 
    st.markdown("---")
    st.markdown("### 📏 Test quality")
    st.metric(
        "F1 Macro",
        f"{info_sel['f1_macro']:.3f}",
        help="Averages the F1 score across all 3 levels equally. Closer to 1.0 is better. Best metric when classes are unbalanced.",
    )
    st.metric(
        "Accuracy",
        f"{info_sel['accuracy']:.3f}",
        help="Percentage of students classified correctly. E.g. 0.939 = 93.9% correct. Can be misleading if one class dominates.",
    )
    st.metric(
        "AUC-ROC",
        f"{info_sel['auc_roc']:.3f}",
        help="Measures how well the model separates the 3 levels. 1.0 = perfect, 0.5 = random. Above 0.85 is very good.",
    )
    st.caption("Evaluated on the 30% of data not used in training.")
 
    with st.expander("📖 Metrics glossary"):
        st.markdown("""
**F1 Macro**
Balances precision and recall across all 3 performance levels equally. Best metric when classes are unbalanced (few Superior students). Ranges from 0 to 1.
 
**Accuracy**
How many students were classified correctly out of the total. Simple but can be misleading — a model that always predicts "Basic" would score ~83% since most students fall in that level.
 
**AUC-ROC**
Measures the model's ability to distinguish between levels. 0.9+ is excellent · 0.7–0.9 is good · below 0.7 needs improvement.
 
**Confidence**
Probability the model assigns to its own prediction. E.g. 99.3% means the model is very certain. Low confidence (< 60%) means the student is near the boundary between two levels.
        """)
 
    with st.expander("📊 Compare all 6 models"):
        filas = []
        for clave, info in info_modelos.items():
            filas.append({
                "Model": nombre_modelo_en(clave),
                "F1 Macro": info["f1_macro"],
                "Accuracy": info["accuracy"],
                "AUC-ROC": info["auc_roc"],
            })
        st.dataframe(
            pd.DataFrame(filas).sort_values("F1 Macro", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
 
    st.markdown("---")
    st.markdown("**Training population**")
    for k, v in metadata["distribucion_target"].items():
        st.text(f"  {nombres_nivel[int(k)]}: {v:,}")
 
# ─── FORM ──────────────────────────────────────────────────────────────────────
opciones = metadata["opciones_formulario"]
 
with st.form("formulario_estudiante"):
    st.markdown("### ✏️ Student data")
    valores_en = {}
 
    # ── Section 1: Academic & Habits ──────────────────────────────────────────
    st.markdown("#### 📐 Academic indicators & habits")
    col1, col2 = st.columns(2)
 
    with col1:
        for col in CAMPOS_ACADEMICO_NUM:
            if col in metadata["features_num"]:
                cfg = opciones[col]
                valores_en[col] = st.number_input(
                    etiqueta_variable(col),
                    min_value=int(cfg["min"]),
                    max_value=int(cfg["max"]),
                    value=int(cfg["default"]),
                )
 
    with col2:
        for col in CAMPOS_HABITOS_NUM:
            if col in metadata["features_num"]:
                cfg = opciones[col]
                valores_en[col] = st.number_input(
                    etiqueta_variable(col),
                    min_value=int(cfg["min"]),
                    max_value=int(cfg["max"]),
                    value=int(cfg["default"]),
                )
        for col in CAMPOS_HABITOS_CAT:
            if col in metadata["features_cat"]:
                cfg = opciones[col]
                opciones_es = [valor_a_es(v) for v in cfg["opciones"]]
                mapa = {valor_a_es(v): v for v in cfg["opciones"]}
                default_es = valor_a_es(cfg["default"])
                idx = opciones_es.index(default_es) if default_es in opciones_es else 0
                sel = st.selectbox(etiqueta_variable(col), opciones_es, index=idx)
                valores_en[col] = mapa[sel]
 
    st.markdown("---")
 
    # ── Section 2: Sociodemographic ───────────────────────────────────────────
    st.markdown("#### 🏠 Sociodemographic context")
    col3, col4 = st.columns(2)
 
    campos_socio_disponibles = [c for c in CAMPOS_SOCIO_CAT if c in metadata["features_cat"]]
    mitad = (len(campos_socio_disponibles) + 1) // 2
 
    with col3:
        for col in campos_socio_disponibles[:mitad]:
            cfg = opciones[col]
            opciones_es = [valor_a_es(v) for v in cfg["opciones"]]
            mapa = {valor_a_es(v): v for v in cfg["opciones"]}
            default_es = valor_a_es(cfg["default"])
            idx = opciones_es.index(default_es) if default_es in opciones_es else 0
            sel = st.selectbox(etiqueta_variable(col), opciones_es, index=idx)
            valores_en[col] = mapa[sel]
 
    with col4:
        for col in campos_socio_disponibles[mitad:]:
            cfg = opciones[col]
            opciones_es = [valor_a_es(v) for v in cfg["opciones"]]
            mapa = {valor_a_es(v): v for v in cfg["opciones"]}
            default_es = valor_a_es(cfg["default"])
            idx = opciones_es.index(default_es) if default_es in opciones_es else 0
            sel = st.selectbox(etiqueta_variable(col), opciones_es, index=idx)
            valores_en[col] = mapa[sel]
 
    # Fallback: any remaining features not yet captured
    for col in metadata["features_num"]:
        if col not in valores_en:
            valores_en[col] = int(opciones[col]["default"])
    for col in metadata["features_cat"]:
        if col not in valores_en:
            valores_en[col] = opciones[col]["default"]
 
    enviado = st.form_submit_button(
        "✨ Calculate performance level",
        type="primary",
        use_container_width=True,
    )
 
# ─── RESULTS ───────────────────────────────────────────────────────────────────
if enviado:
    X_nuevo = construir_entrada(valores_en)
    predicciones = {}
    probabilidades = {}
 
    for label, clave in OPCIONES_MODELO.items():
        pipe = pipelines[clave]
        pred = int(pipe.predict(X_nuevo)[0])
        prob = pipe.predict_proba(X_nuevo)[0]
        predicciones[clave] = pred
        probabilidades[clave] = prob
 
    pred_sel = predicciones[clave_sel]
    prob_sel = probabilidades[clave_sel]
 
    st.divider()
 
    # ── Detail first ──────────────────────────────────────────────────────────
    st.markdown(f"### 🔎 Detail: {modelo_sel_label}")
 
    c1, c2 = st.columns([1, 2])
    with c1:
        emoji_sel = ["🔴", "🟠", "🟢"][pred_sel]
        st.metric(
            "Estimated level",
            f"{emoji_sel} {nombres_nivel[pred_sel]}",
            help=f"Output of the {modelo_sel_label} model.",
        )
        confianza = float(prob_sel[pred_sel])
        st.metric(
            "Model confidence",
            f"{confianza * 100:.1f} %",
            help="Probability the model assigns to its own prediction. Above 80% is high confidence.",
        )
        alerta_nivel(pred_sel)
 
    with c2:
        st.plotly_chart(
            grafico_probabilidades(prob_sel, f"Probability by level — {modelo_sel_label}"),
            use_container_width=True,
        )
 
    pipeline_sel = pipelines[clave_sel]
    fig_imp = grafico_importancia(pipeline_sel)
    if fig_imp is not None:
        with st.expander("🔬 Which factors influence this prediction the most?"):
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Shows how much each variable contributes to the tree-based model. "
                "Does not indicate whether the variable increases or decreases the score by itself."
            )
    else:
        with st.expander("🔬 How to interpret this model"):
            if clave_sel in ("rl", "rl_smote"):
                st.info(
                    "Logistic regression combines all encoded variables linearly. "
                    "See the project notebook for detailed coefficients."
                )
            elif clave_sel == "pca_km":
                st.info(
                    "This method groups study patterns and associates each cluster "
                    "with the most frequent performance level in that group."
                )
 
    mejor_clave = metadata["mejor_modelo"]
    mejor = info_modelos[mejor_clave]
    rl = info_modelos["rl"]
 
    with st.expander("💡 Which model to use in production?"):
        st.markdown(
            f"""
**Recommended:** **{nombre_modelo_en(mejor_clave)}**  
F1 Macro = {mejor['f1_macro']:.3f} · Accuracy = {mejor['accuracy']:.3f} · AUC-ROC = {mejor['auc_roc']:.3f}
 
In section 13 of the project, **Gradient Boosting** captured non-linear relationships between attendance, study hours and family context better than other models.
 
**Logistic Regression** (F1 Macro = {rl['f1_macro']:.3f}): very precise on the **Basic** class, but detects extreme levels less often due to class imbalance.
 
Use the other models in this app to **compare** whether the predicted level changes across algorithms.
            """
        )
 
    st.divider()
 
    # ── All models ────────────────────────────────────────────────────────────
    st.markdown("### 🔀 Results across all models")
 
    cols = st.columns(3)
    for i, (label, clave) in enumerate(OPCIONES_MODELO.items()):
        with cols[i % 3]:
            pred = predicciones[clave]
            info = info_modelos[clave]
            emoji_nivel = ["🔴", "🟠", "🟢"][pred]
            st.markdown(f"**{label}**")
            st.metric("Level", f"{emoji_nivel} {nombres_nivel[pred]}")
            st.caption(
                f"F1 Macro: {info['f1_macro']:.3f} · Accuracy: {info['accuracy']:.3f}"
            )
            if clave == clave_sel:
                st.caption("★ Selected model in sidebar")
 
