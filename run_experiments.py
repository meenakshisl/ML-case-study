import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import (
    read_dataset,
    split_features_labels,
    build_models,
    build_pipelines,
    run_experiment,
    make_barplot,
    plot_confusion_matrix,
    stratified_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-flip poisoning experiments on malware detection data.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV (features + label).")
    parser.add_argument("--label-col", default=None, help="Name of the label column (binary).")
    parser.add_argument("--sample-size", type=int, default=10000, help="Stratified sample size for the demo.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction for train/test split.")
    parser.add_argument("--flip-fracs", type=float, nargs="+", default=[0.0, 0.1, 0.2], help="Flip fractions to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="ML-case-study/outputs", help="Directory to write outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading dataset: {args.input_csv}")
    df, label_col = read_dataset(args.input_csv, label_col=args.label_col)
    print(f"Label column: {label_col}")
    if args.sample_size > 0:
        df = stratified_sample(df, label_col=label_col, sample_size=args.sample_size, seed=args.seed)
        print(f"Sampled dataset to {len(df)} rows (stratified).")

    X, y, num_cols, cat_cols = split_features_labels(df, label_col=label_col, drop_non_numeric=True)
    print(f"Numeric features: {len(num_cols)}; Categorical kept: {len(cat_cols)}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values, test_size=args.test_size, random_state=args.seed, stratify=y.values
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    models = build_models(random_state=args.seed)
    pipelines = build_pipelines(models, numeric_cols=num_cols, categorical_cols=cat_cols)

    print(f"Running experiments for flip fractions: {args.flip_fracs}")
    results, cms = run_experiment(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        pipelines=pipelines, flip_fracs=args.flip_fracs, seed=args.seed,
    )
    results_path = os.path.join(args.output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    print(f"Wrote results to {results_path}")

    # Plot accuracy barplot
    barplot_path = os.path.join(args.output_dir, "metrics_barplot.png")
    make_barplot(results, output_path=barplot_path)
    print(f"Wrote bar plot to {barplot_path}")

    # Save a few confusion matrices for the highest flip fraction
    max_frac = max(args.flip_fracs)
    chosen_models = ["LogisticRegression", "RandomForest", "MLP"]
    for m in chosen_models:
        key = (m, max_frac)
        if key in cms:
            out = os.path.join(args.output_dir, f"cm_{m}_{str(max_frac).replace('.', '')}.png")
            plot_confusion_matrix(cms[key], class_labels=["Benign", "Malware"], title=f"{m} @ flip={max_frac}", output_path=out)
            print(f"Wrote confusion matrix: {out}")

    print("Done.")


if __name__ == "__main__":
    main()


