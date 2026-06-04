"""
Entrena los 6 modelos multiclase de la sección 13 (Target Multiclase).
Ejecutar UNA VEZ con: python entrenamiento.py
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

from pca_kmeans_classifier import PCAKMeansClassifier

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "StudentPerformance.csv"

NIVELES = {0: "Deficiente", 1: "Básico", 2: "Superior"}
BINS = [-1, 64, 74, 100]
LABELS = [0, 1, 2]

MODELOS_ARCHIVO = {
    "rl": "modelo_regresion_logistica.pkl",
    "rl_smote": "modelo_rl_smote.pkl",
    "tree": "modelo_arbol.pkl",
    "rf": "modelo_random_forest.pkl",
    "gb": "modelo_gradient_boosting.pkl",
    "pca_km": "modelo_pca_kmeans.pkl",
}


def cargar_y_preparar() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    data = df.copy()
    data = data.dropna()
    data = data[data["Exam_Score"] <= 100].copy()
    data["Total_Engagement"] = (data["Attendance"] / 100) * data["Hours_Studied"]
    data["Study_Efficiency"] = data["Exam_Score"] / data["Hours_Studied"].replace(0, 1)
    data["Target"] = pd.cut(
        data["Exam_Score"], bins=BINS, labels=LABELS
    ).astype(int)

    X = data.drop(columns=["Target", "Exam_Score", "Study_Efficiency", "Total_Engagement"])
    y = data["Target"]
    return X, y, data


def crear_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    ), numeric_features, categorical_features


def metricas_test(modelo, X_test, y_test) -> dict:
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)
    clases = [0, 1, 2]
    y_bin = label_binarize(y_test, classes=clases)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro"), 4),
        "auc_roc": round(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"), 4
        ),
    }


def main():
    print("Cargando datos...")
    X, y, data = cargar_y_preparar()
    preprocessor, features_num, features_cat = crear_preprocesador(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    config = [
        (
            "Regresión Logística",
            "rl",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.1, 1, 10]},
        ),
        (
            "RL + SMOTE",
            "rl_smote",
            ImbPipeline(
                [
                    ("prep", clone(preprocessor)),
                    ("smote", SMOTE(random_state=42)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.1, 1, 10]},
        ),
        (
            "Árbol de Decisión",
            "tree",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    (
                        "clf",
                        DecisionTreeClassifier(
                            random_state=42, class_weight="balanced"
                        ),
                    ),
                ]
            ),
            {
                "clf__max_depth": [5, 10, 15, None],
                "clf__min_samples_leaf": [1, 5, 10],
            },
        ),
        (
            "Random Forest",
            "rf",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    (
                        "clf",
                        RandomForestClassifier(
                            random_state=42, class_weight="balanced"
                        ),
                    ),
                ]
            ),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [10, 15, None]},
        ),
        (
            "Gradient Boosting",
            "gb",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    ("clf", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
        (
            "PCA + K-Means",
            "pca_km",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    ("clf", PCAKMeansClassifier(random_state=42)),
                ]
            ),
            {"clf__n_components": [5, 10, 15], "clf__n_clusters": [3, 6, 9]},
        ),
    ]

    os.makedirs(MODELS_DIR, exist_ok=True)
    resultados = {}

    print("=" * 55)
    print("GRIDSEARCH — F1-macro (validación cruzada)")
    print("=" * 55)

    for nombre, clave, pipeline, params in config:
        print(f"\nEntrenando: {nombre}...")
        gs = GridSearchCV(
            pipeline, params, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        test_m = metricas_test(best, X_test, y_test)

        ruta = MODELS_DIR / MODELOS_ARCHIVO[clave]
        joblib.dump(best, ruta)

        resultados[clave] = {
            "nombre": nombre,
            "archivo": str(ruta.name),
            "f1_cv": round(gs.best_score_, 4),
            "best_params": gs.best_params_,
            **test_m,
        }
        print(f"  F1-macro CV: {gs.best_score_:.4f} | params: {gs.best_params_}")
        print(
            f"  Test - F1-macro: {test_m['f1_macro']:.4f} | "
            f"Accuracy: {test_m['accuracy']:.4f} | AUC: {test_m['auc_roc']:.4f}"
        )

    mejor = max(resultados, key=lambda k: resultados[k]["f1_macro"])
    print("\n" + "=" * 55)
    print(
        f"Mejor modelo (F1-macro test): {resultados[mejor]['nombre']} "
        f"({resultados[mejor]['f1_macro']:.4f})"
    )
    print("=" * 55)

    opciones_formulario = {}
    for col in features_num:
        opciones_formulario[col] = {
            "tipo": "numero",
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "default": float(X[col].median()),
        }
    for col in features_cat:
        valores = sorted(X[col].dropna().astype(str).unique().tolist())
        opciones_formulario[col] = {
            "tipo": "categoria",
            "opciones": valores,
            "default": valores[0],
        }

    metadata = {
        "niveles": NIVELES,
        "bins_exam_score": BINS,
        "features_num": features_num,
        "features_cat": features_cat,
        "columnas_orden": list(X.columns),
        "opciones_formulario": opciones_formulario,
        "modelos": resultados,
        "mejor_modelo": mejor,
        "distribucion_target": data["Target"].value_counts().sort_index().to_dict(),
    }
    joblib.dump(metadata, MODELS_DIR / "metadata.pkl")
    print(f"\nOK: Modelos y metadata en {MODELS_DIR}/")


if __name__ == "__main__":
    main()
