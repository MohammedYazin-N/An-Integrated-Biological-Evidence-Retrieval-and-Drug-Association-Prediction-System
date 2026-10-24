# BIO AI: Evidence-Guided Disease–Gene–Variant–Drug Association Retrieval and ML-Based Candidate Ranking

An integrated BIO AI system that combines biological evidence retrieval with machine learning-based disease–gene–variant–drug association prediction and candidate ranking, to support biomedical research and drug candidate prioritization.

**Live app:** https://bio-ai-project.streamlit.app/

---

## Scientific note (please read)

- The only dataset used is `ml_model_features.csv`.
- `Association_Label` = 1 means the combination was a **positive example** when the dataset was built. 0 means a **negative example**.
- **A label of 0 does not mean a drug is ineffective.** The positive rate is exactly 25% in every disease and in both the train and test splits, which suggests the negatives were generated for machine-learning purposes rather than confirmed one by one.
- A model score shows **statistical similarity to the dataset's positive examples**. It is not clinical advice, a measure of drug effectiveness, a probability of cure, or an FDA approval status.
- Records retrieved from the dataset are not independently verified clinical evidence.

---

## What the app does

The app is built in two layers that are kept separate on purpose.

**Layer 1: Evidence Retrieval.** Search existing dataset records by Disease, Gene, Variant or Drug, individually or in combination. It supports dropdown search, partial-match text search, label filtering and CSV export.

**Layer 2: ML Prediction and Ranking.** A trained model scores how similar a Disease–Gene–Variant–Drug combination is to the dataset's positive examples, and ranks several candidates by that score. Predictions are never added to the evidence records.

### Pages

| Page | Purpose |
|---|---|
| Overview | Dataset summary and charts |
| Evidence Search | Look up existing records |
| ML Prediction | Score one combination |
| Candidate Ranking | Score and rank many combinations |
| Evidence Comparison | Prediction next to matching dataset records |
| Model Information | Metrics, explainability, metadata, limitations |

---

## Dataset

| Item | Value |
|---|---|
| Records | 5,468 |
| Diseases | 42 |
| Genes | 348 |
| Variants | 667 |
| Drugs | 164 |
| Positive labels | 25% |

Train/test split: 4,308 / 1,160 rows. The dataset's own `Split` column was used. No Gene+Variant group appears in both train and test, so related records do not leak between them.

Model inputs: `Disease`, `Gene`, `Variant`, `Drug`. Target: `Association_Label`. `Row_ID`, `Group_ID` and `Split` are not used as predictors.

---

## Model and results

Five models were trained and compared on the held-out test set. All use a one-hot encoder inside a scikit-learn pipeline, so the saved model applies the same preprocessing as training.

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Baseline (most frequent) | 0.750 | 0.000 | 0.500 | 0.250 |
| Logistic Regression | 0.776 | 0.395 | 0.732 | 0.503 |
| Random Forest | 0.823 | 0.565 | 0.817 | 0.687 |
| **Extra Trees (selected)** | **0.849** | **0.659** | **0.820** | **0.713** |
| Gradient Boosting | 0.799 | 0.404 | 0.744 | 0.547 |

The model was selected by test-set ROC-AUC. The baseline's 0.75 accuracy comes from always predicting the negative label, which is why ROC-AUC and PR-AUC are reported too.

### Explainability

Permutation feature importance (drop in ROC-AUC when a field is shuffled):

| Feature | Importance |
|---|---|
| Drug | 0.313 |
| Disease | 0.087 |
| Gene | 0.069 |
| Variant | 0.003 |

The model relies mostly on the drug name. Variant contributes almost nothing, so it is not learning variant-specific patterns. This describes a statistical pattern in this dataset, not a biological cause.

---

## Limitations

- Trained on one dataset and evaluated on one train/test split. Performance may not generalize.
- High scores do not prove clinical effectiveness, safety or approval.
- Values never seen in training (a new drug, gene, disease or variant) carry no learned information. The app warns when you enter one.
- Disease and drug names are stored in lowercase in the dataset. The app lowercases those two fields before predicting.
- Frequency in the dataset reflects how the data was collected, not biological importance.

---

## Repository contents

| File | Description |
|---|---|
| `app.py` | Streamlit application |
| `ml_model_features.csv` | Source dataset |
| `trained_bioai_model.joblib` | Saved preprocessing + model pipeline |
| `model_metadata.json` | Target definition, features, metrics, limitations |
| `model_evaluation_report.csv` | Metrics for all five models |
| `explainability_permutation_importance.csv` | Feature importance results |
| `requirements.txt` | Pinned dependencies |
| `.streamlit/config.toml` | Dark theme |

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows, if `pip` or `streamlit` is not recognised, use:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The saved model only loads with the same scikit-learn version that created it, so keep the pinned versions in `requirements.txt` (scikit-learn 1.8.0).

---

## Team Behind the Project 

- Mohammed Yazin N, MSc Data Science and Bio AI
- Arun PS, MSc Data Science and Bio AI
- Archana T, MSc Data Science and Bio AI
- Nihala Thajudheen, MSc Data Science and Bio AI
