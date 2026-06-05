# 🎯 Predictor de Desempeño Académico

Aplicación de machine learning para estimar el nivel de rendimiento académico de un estudiante a partir de sus características de perfil, contexto y hábitos de estudio.

---

## 📋 Descripción

La app utiliza **6 modelos de clasificación multiclase** entrenados sobre el dataset `StudentPerformance.csv`. A partir de los datos ingresados, predice si el estudiante quedará en uno de tres niveles:

| Nivel | Rango de nota | Indicador |
|---|---|---|
| 🔴 Deficiente | ≤ 64 | Requiere plan de refuerzo |
| 🟠 Básico | 65 – 74 | Rendimiento intermedio |
| 🟢 Superior | ≥ 75 | Buen desempeño proyectado |

---

## 🧠 Modelos incluidos

| Modelo | Clave interna | Archivo |
|---|---|---|
| Potenciación de gradiente ⭐ | `gb` | `modelo_gradient_boosting.pkl` |
| Regresión logística | `rl` | `modelo_regresion_logistica.pkl` |
| Regresión logística + SMOTE | `rl_smote` | `modelo_rl_smote.pkl` |
| Bosque aleatorio | `rf` | `modelo_random_forest.pkl` |
| Árbol de decisión | `tree` | `modelo_arbol.pkl` |
| PCA + K-Means | `pca_km` | `modelo_pca_kmeans.pkl` |

> ⭐ El modelo recomendado es **Potenciación de gradiente**, por obtener el mejor F1-macro en datos de prueba.

---

## 📁 Estructura de archivos

```
rendimiento-estudiantes-streamlit/
│
├── app_ml.py                   # App de predicción (este módulo)
├── entrenamiento.py            # Script para re-entrenar los modelos
├── pca_kmeans_classifier.py    # Clasificador personalizado PCA + K-Means
├── ui_tema.py                  # Componentes visuales y tema compartido
│
├── data/
│   └── StudentPerformance.csv  # Dataset de entrenamiento
│
└── models/
    ├── modelo_gradient_boosting.pkl
    ├── modelo_regresion_logistica.pkl
    ├── modelo_rl_smote.pkl
    ├── modelo_random_forest.pkl
    ├── modelo_arbol.pkl
    ├── modelo_pca_kmeans.pkl
    └── metadata.pkl            # Métricas, columnas y opciones del formulario
```

---

## ⚙️ Requisitos

```txt
streamlit
pandas
numpy
scikit-learn==1.5.2
plotly
joblib
imbalanced-learn==0.12.4
scipy
statsmodels
```

> **Python requerido: 3.11**  
> El archivo `.python-version` en la raíz del repositorio ya especifica esta versión para Streamlit Cloud.

---

## 🚀 Ejecución local

### 1. Instalar dependencias

```bash
pip install scikit-learn==1.5.2 imbalanced-learn==0.12.4 streamlit pandas numpy plotly joblib scipy statsmodels
```

### 2. Entrenar los modelos (solo la primera vez o si cambia el dataset)

```bash
python entrenamiento.py
```

Esto genera todos los archivos `.pkl` en la carpeta `models/`.

### 3. Lanzar la app

```bash
streamlit run app_ml.py
```

---

## 🔄 Re-entrenamiento

Si necesitas actualizar los modelos (nuevo dataset o cambio de parámetros):

```bash
python entrenamiento.py
git add models/
git commit -m "Actualizar modelos entrenados"
git push
```

> ⚠️ Los archivos `.pkl` deben generarse con **las mismas versiones** de `scikit-learn` e `imbalanced-learn` que usa Streamlit Cloud (`sklearn==1.5.2`, `imblearn==0.12.4`), de lo contrario la app fallará al cargarlos.

---

## 📊 Variables de entrada

El formulario solicita las siguientes variables del estudiante:

**Indicadores numéricos**
- Horas de estudio semanales
- Porcentaje de asistencia
- Notas anteriores
- Horas de actividades extracurriculares
- Horas de sueño por día

**Perfil y contexto**
- Género
- Nivel de motivación (Bajo / Medio / Alto)
- Acceso a internet (Sí / No)
- Tipo de colegio (Público / Privado)
- Apoyo familiar (Bajo / Medio / Alto)

---

## 📈 Interpretación de resultados

- La app muestra la predicción de los **6 modelos simultáneamente** para comparar.
- El modelo seleccionado en la barra lateral ofrece detalle completo: nivel estimado, confianza (probabilidad) y gráfico de probabilidades por clase.
- Los modelos basados en árboles (Random Forest, Gradient Boosting, Árbol de decisión) incluyen además un gráfico de **importancia de variables**.

---

## 🗂️ Proyecto

Desarrollado como parte del **Proyecto CDP · Sección 13 — Clasificación Multiclase**.
