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

OPCIONES_MODELO = {
    "Potenciación de gradiente": "gb",
    "Regresión logística": "rl",
    "Regresión logística + SMOTE": "rl_smote",
    "Bosque aleatorio": "rf",
    "Árbol de decisión": "tree",
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

st.set_page_config(
    page_title="Predictor de desempeño estudiantil",
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
            marker=dict(
                color=colores,
                line=dict(color="white", width=1),
            ),
            text=[f"{p * 100:.1f} %" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Probabilidad estimada",
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
        title="Variables con mayor peso en la predicción (top 12)",
        height=440,
        xaxis_title="Importancia relativa",
    )
    return estilo_plotly(fig)


def nombre_modelo_es(clave: str) -> str:
    for etiqueta, k in OPCIONES_MODELO.items():
        if k == clave:
            return etiqueta
    return clave


hero(
    "Predictor de desempeño académico",
    "Estima si el estudiante quedará en nivel **Deficiente** (≤ 64), **Básico** (65–74) o **Superior** (≥ 75). "
    "Incluye los 6 modelos de la sección 13; el recomendado es **Potenciación de gradiente**.",
    "🎯",
)

with st.sidebar:
    st.markdown("### 🧠 Modelo de referencia")
    modelo_sel_label = st.selectbox(
        "Ver detalle e interpretación con",
        list(OPCIONES_MODELO.keys()),
        index=0,
    )
    clave_sel = OPCIONES_MODELO[modelo_sel_label]
    info_sel = info_modelos[clave_sel]

    st.markdown("---")
    st.markdown("### 📏 Calidad en prueba")
    st.metric("F1 macro", f"{info_sel['f1_macro']:.3f}")
    st.metric("Exactitud", f"{info_sel['accuracy']:.3f}")
    st.metric("AUC-ROC", f"{info_sel['auc_roc']:.3f}")
    st.caption("Evaluado sobre el 30 % de datos no usados en entrenamiento.")

    with st.expander("📊 Comparar los 6 modelos"):
        filas = []
        for clave, info in info_modelos.items():
            filas.append(
                {
                    "Modelo": nombre_modelo_es(clave),
                    "F1 macro": info["f1_macro"],
                    "Exactitud": info["accuracy"],
                    "AUC-ROC": info["auc_roc"],
                }
            )
        st.dataframe(
            pd.DataFrame(filas).sort_values("F1 macro", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("**Población de entrenamiento**")
    for k, v in metadata["distribucion_target"].items():
        st.text(f"  {nombres_nivel[int(k)]}: {v:,}")

opciones = metadata["opciones_formulario"]

with st.form("formulario_estudiante"):
    st.markdown("### ✏️ Datos del estudiante")

    col1, col2 = st.columns(2)
    valores_en = {}

    with col1:
        st.markdown("**📐 Indicadores numéricos**")
        for col in metadata["features_num"]:
            cfg = opciones[col]
            valores_en[col] = st.number_input(
                etiqueta_variable(col),
                min_value=int(cfg["min"]),
                max_value=int(cfg["max"]),
                value=int(cfg["default"]),
            )

    with col2:
        st.markdown("**🏷️ Perfil y contexto**")
        for col in metadata["features_cat"]:
            cfg = opciones[col]
            opciones_es = [valor_a_es(v) for v in cfg["opciones"]]
            mapa_display = {valor_a_es(v): v for v in cfg["opciones"]}
            default_es = valor_a_es(cfg["default"])
            idx = opciones_es.index(default_es) if default_es in opciones_es else 0
            seleccion_es = st.selectbox(
                etiqueta_variable(col),
                opciones_es,
                index=idx,
            )
            valores_en[col] = mapa_display[seleccion_es]

    enviado = st.form_submit_button(
        "✨ Calcular nivel de desempeño",
        type="primary",
        use_container_width=True,
    )

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
    st.markdown("### 🔀 Resultados por modelo")

    cols = st.columns(3)
    for i, (label, clave) in enumerate(OPCIONES_MODELO.items()):
        with cols[i % 3]:
            pred = predicciones[clave]
            info = info_modelos[clave]
            emoji_nivel = ["🔴", "🟠", "🟢"][pred]
            st.markdown(f"**{label}**")
            st.metric("Nivel", f"{emoji_nivel} {nombres_nivel[pred]}")
            st.caption(
                f"F1 macro: {info['f1_macro']:.3f} · Exactitud: {info['accuracy']:.3f}"
            )
            if clave == clave_sel:
                st.caption("★ Modelo seleccionado en la barra lateral")

    st.divider()
    st.markdown(f"### 🔎 Detalle: {modelo_sel_label}")

    c1, c2 = st.columns([1, 2])
    with c1:
        emoji_sel = ["🔴", "🟠", "🟢"][pred_sel]
        st.metric(
            "Nivel estimado",
            f"{emoji_sel} {nombres_nivel[pred_sel]}",
            help=f"Salida del modelo {modelo_sel_label}.",
        )
        confianza = float(prob_sel[pred_sel])
        st.metric("Confianza del modelo", f"{confianza * 100:.1f} %")
        alerta_nivel(pred_sel)

    with c2:
        st.plotly_chart(
            grafico_probabilidades(
                prob_sel,
                f"Probabilidad por nivel — {modelo_sel_label}",
            ),
            use_container_width=True,
        )

    pipeline_sel = pipelines[clave_sel]
    fig_imp = grafico_importancia(pipeline_sel)
    if fig_imp is not None:
        with st.expander("🔬 ¿Qué factores pesan más en esta predicción?"):
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Mide cuánto contribuye cada variable al modelo de árboles. "
                "No indica si sube o baja la nota por sí sola."
            )
    else:
        with st.expander("🔬 Cómo interpretar este modelo"):
            if clave_sel in ("rl", "rl_smote"):
                st.info(
                    "La regresión logística combina todas las variables codificadas. "
                    "Consulta el notebook del proyecto para ver coeficientes detallados."
                )
            elif clave_sel == "pca_km":
                st.info(
                    "Este método agrupa patrones de estudio y los asocia al nivel "
                    "de desempeño más frecuente en cada grupo."
                )

    mejor_clave = metadata["mejor_modelo"]
    mejor = info_modelos[mejor_clave]
    rl = info_modelos["rl"]

    with st.expander("💡 ¿Qué modelo usar en producción?"):
        st.markdown(
            f"""
**Recomendado:** **{nombre_modelo_es(mejor_clave)}**  
F1 macro = {mejor['f1_macro']:.3f} · Exactitud = {mejor['accuracy']:.3f} · AUC-ROC = {mejor['auc_roc']:.3f}

En la sección 13 del proyecto, la **potenciación de gradiente** capturó mejor las relaciones
no lineales entre asistencia, estudio y contexto familiar.

**Regresión logística** (F1 macro = {rl['f1_macro']:.3f}): muy precisa en la clase **Básico**,
pero detecta con menos frecuencia los niveles extremos por el desbalance de clases.

Usa los otros modelos en esta app para **comparar** si el nivel cambia según el algoritmo.
            """
        )
