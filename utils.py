from __future__ import annotations

import os
import math
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def infer_label_column(columns: Iterable[str]) -> Optional[str]:
    candidates = [
        "label", "Label", "target", "Target", "class", "Class",
        "malware", "Malware", "is_malware", "IsMalware",
        "HasDetections", "hasDetections", "has_detections",
    ]
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def ensure_binary_labels(y: pd.Series) -> pd.Series:
    # Convert boolean to int
    if y.dtype == bool:
        return y.astype(int)
    # Many datasets use 0/1 ints already
    uniques = sorted(pd.unique(y))
    if len(uniques) == 2:
        # Map the smaller value to 0 and the larger to 1
        mapping = {uniques[0]: 0, uniques[1]: 1}
        return y.map(mapping).astype(int)
    raise ValueError(f"Expected binary labels, found {len(uniques)} unique values: {uniques[:10]}")


def read_dataset(csv_path: str, label_col: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if label_col is None:
        label_col = infer_label_column(df.columns)
        if label_col is None:
            raise ValueError("Could not infer label column. Please provide --label-col.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in CSV columns.")
    return df, label_col


def split_features_labels(
    df: pd.DataFrame,
    label_col: str,
    drop_non_numeric: bool = True,
    low_cardinality_threshold: int = 10
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y_raw = df[label_col]
    y = ensure_binary_labels(y_raw)
    X = df.drop(columns=[label_col])

    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if drop_non_numeric:
        # Drop high-cardinality non-numeric columns entirely
        categorical_cols = []
        X = X[numeric_cols]
    else:
        # Keep low-cardinality categoricals; drop high-cardinality categoricals
        kept = []
        for c in categorical_cols:
            nunique = X[c].nunique(dropna=True)
            if nunique <= low_cardinality_threshold:
                kept.append(c)
        categorical_cols = kept
        drop_cols = [c for c in X.columns if (c not in numeric_cols) and (c not in categorical_cols)]
        if drop_cols:
            X = X.drop(columns=drop_cols)

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: List[Tuple[str, BaseEstimator]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_tf = Pipeline(numeric_steps)

    if categorical_cols:
        categorical_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    else:
        categorical_tf = "drop"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    return {
        "SGD": SGDClassifier(random_state=random_state, max_iter=1000, tol=1e-3),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LinearSVM": LinearSVC(random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "Perceptron": Perceptron(random_state=random_state, max_iter=1000),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, early_stopping=True, random_state=random_state),
    }


def needs_scaling(model_name: str) -> bool:
    return model_name in {"SGD", "LogisticRegression", "KNN", "LinearSVM", "Perceptron", "MLP"}


def build_pipelines(
    models: Dict[str, BaseEstimator],
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Dict[str, Pipeline]:
    pipelines: Dict[str, Pipeline] = {}
    for name, model in models.items():
        pre = build_preprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            scale_numeric=needs_scaling(name),
        )
        pipelines[name] = Pipeline([
            ("pre", pre),
            ("clf", model),
        ])
    return pipelines


def flip_labels(y: np.ndarray, flip_fraction: float, rng_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if flip_fraction <= 0.0:
        return y.copy(), np.array([], dtype=int)
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    k = int(flip_fraction * n)
    idx = rng.choice(n, size=k, replace=False)
    y_poison = y.copy()
    y_poison[idx] = 1 - y_poison[idx]
    return y_poison, idx


def train_and_eval(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, zero_division=0),
        "rec": recall_score(y_test, y_pred, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred),
    }


def run_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    pipelines: Dict[str, Pipeline],
    flip_fracs: Iterable[float],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, float], np.ndarray]]:
    rows: List[Dict[str, object]] = []
    cms: Dict[Tuple[str, float], np.ndarray] = {}
    for frac in flip_fracs:
        y_train_poison, _ = flip_labels(y_train, flip_fraction=frac, rng_seed=seed)
        for name, pipe in pipelines.items():
            metrics = train_and_eval(pipe, X_train, y_train_poison, X_test, y_test)
            cm = metrics.pop("cm")
            rows.append({
                "model": name,
                "flip_frac": frac,
                **metrics,
            })
            cms[(name, frac)] = cm
    results = pd.DataFrame(rows).sort_values(["model", "flip_frac"]).reset_index(drop=True)
    return results, cms


def make_barplot(results: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    pivot = results.pivot(index="model", columns="flip_frac", values="acc")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Accuracy by Model and Flip Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.legend(title="Flip Fraction")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def make_barplot_metric(results: pd.DataFrame, metric: str, output_path: Optional[str] = None) -> plt.Figure:
    if metric not in {"acc", "prec", "rec"}:
        raise ValueError(f"Unsupported metric '{metric}'. Choose one of: acc, prec, rec")
    pivot = results.pivot(index="model", columns="flip_frac", values=metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    title_map = {"acc": "Accuracy", "prec": "Precision", "rec": "Recall"}
    ax.set_title(f"{title_map[metric]} by Model and Flip Fraction")
    ax.set_ylabel(title_map[metric])
    ax.set_xlabel("Model")
    ax.legend(title="Flip Fraction")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_labels: List[str], title: str, output_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def stratified_sample(df: pd.DataFrame, label_col: str, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df
    # Sample within each class proportionally
    rng = np.random.default_rng(seed)
    parts = []
    for label_value, group in df.groupby(label_col):
        frac = len(group) / len(df)
        k = max(1, int(round(frac * sample_size)))
        idx = rng.choice(len(group), size=min(k, len(group)), replace=False)
        parts.append(group.iloc[idx])
    sampled = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


