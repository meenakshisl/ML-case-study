# Label‑Flip Poisoning on Malware Detection (Reproduction)

This repo reproduces the core methodology from:
- Aryal et al., “Analysis of Label‑Flip Poisoning Attack on Machine Learning Based Malware Detector” — http://arxiv.org/pdf/2301.01044

We train 8 classic ML models and evaluate how randomly flipping a fraction of training labels (10–20%+) impacts Accuracy, Precision, and Recall, while keeping the test set clean.

## Features
- 8 classic ML models via unified `scikit-learn` pipelines (scaling where appropriate)
- CLI to reproduce experiments and generated figures
- Notebook for interactive exploration
- Markdown slides with export to editable PPTX (via Pandoc)

## Repo layout
- `poison_label_flip_demo.ipynb` — interactive notebook.
- `run_experiments.py` — CLI runner for reproducible experiments.
- `utils.py` — helpers (data prep, model zoo, evaluation, plotting).
- `slides/poisoning_label_flip_slides.md` — talk deck (Markdown).
- `requirements.txt` — Python dependencies.
- `outputs/` — generated metrics and figures (created at run time).

## Data
Use your own tabular binary malware dataset (or a proxy) with a label column.
- Common label names: `legitimate`, `HasDetections`, `label`, `target` (see auto‑inference in `utils.py`).
- This repo does not redistribute third‑party datasets; please honor original licenses/ToS.

## Environment
Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ML-case-study/requirements.txt
```

## Reproduce (CLI)
Run on your CSV (update paths/label as needed):

```bash
python ML-case-study/run_experiments.py \
  --input-csv /path/to/your_data.csv \
  --label-col legitimate \
  --sample-size 10000 \
  --test-size 0.2 \
  --flip-fracs 0.0 0.1 0.2 \
  --seed 42 \
  --output-dir ML-case-study/outputs
```

Outputs:
- `outputs/results.csv` — metrics per model × flip fraction.
- `outputs/metrics_barplot.png` — accuracy vs. flip fraction.
- `outputs/cm_{Model}_{frac}.png` — confusion matrices at highest flip fraction.

## Models
Eight models (per paper):
- SGD, RandomForest, LogisticRegression, KNN, LinearSVM, DecisionTree, Perceptron, MLP

## Method summary
- Stratified train/test split (default 80/20)
- Baseline on clean labels, then re‑train with label flips at specified fractions
- Test set remains unchanged (clean) for fair comparison
- Metrics: Accuracy, Precision, Recall; selected confusion matrices

## Slides → editable PPTX (Pandoc)
1) (Optional) Create a custom PowerPoint theme and save as:
   `ML-case-study/slides/reference_theme.pptx`
2) Export:
```bash
pandoc ML-case-study/slides/poisoning_label_flip_slides.md \
  -o ML-case-study/outputs/label_flip_demo.pptx \
  --from markdown+yaml_metadata_block \
  --slide-level=2 \
  --reference-doc=ML-case-study/slides/reference_theme.pptx \
  --resource-path=".:ML-case-study/slides:ML-case-study/outputs"
```
If you skip `--reference-doc`, Pandoc produces a default editable PPTX.

## Notebook (optional)
Open and run:
```bash
jupyter notebook ML-case-study/poison_label_flip_demo.ipynb
```

## Citation
If you use this code in academic work, please cite the original paper:
- Aryal et al., “Analysis of Label‑Flip Poisoning Attack on Machine Learning Based Malware Detector” — http://arxiv.org/pdf/2301.01044

## License
Specify your project license (e.g., MIT). If contributing, include standard Contributor guidelines.

## Acknowledgments
- Thanks to the authors of the referenced paper and the scikit‑learn ecosystem.
