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
| Gradient Boosting ⭐ | `gb` | `modelo_gradient_boosting.pkl` |
| Logistic Regression | `rl` | `modelo_regresion_logistica.pkl` |
| Logistic Regression + SMOTE | `rl_smote` | `modelo_rl_smote.pkl` |
| Random Forest | `rf` | `modelo_random_forest.pkl` |
| Decision Tree | `tree` | `modelo_arbol.pkl` |
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

