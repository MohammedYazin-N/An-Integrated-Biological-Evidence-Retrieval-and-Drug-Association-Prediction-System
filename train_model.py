import pandas as pd
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.inspection import permutation_importance

RANDOM_SEED = 42
INPUT_FEATURES = ["Disease", "Gene", "Variant", "Drug"]
TARGET_COLUMN = "Association_Label"

print("Loading dataset...")
data = pd.read_csv("ml_model_features.csv")

train_data = data[data["Split"] == "train"].copy()
test_data = data[data["Split"] == "test"].copy()
X_train, y_train = train_data[INPUT_FEATURES], train_data[TARGET_COLUMN]
X_test, y_test = test_data[INPUT_FEATURES], test_data[TARGET_COLUMN]

def build_pipeline(model):
    preprocessing = ColumnTransformer([
        ("encode_categories", OneHotEncoder(handle_unknown="ignore"), INPUT_FEATURES)
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", model)])

candidate_models = {
    "baseline_most_frequent": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
    "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
}

fitted_pipelines = {}
all_metrics = []
for name, model in candidate_models.items():
    print(f"Training {name} ...")
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)
    fitted_pipelines[name] = pipeline

    predicted_labels = pipeline.predict(X_test)
    predicted_scores = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    all_metrics.append({
        "model_name": name,
        "accuracy": accuracy_score(y_test, predicted_labels),
        "precision": precision_score(y_test, predicted_labels, zero_division=0),
        "recall": recall_score(y_test, predicted_labels, zero_division=0),
        "f1_score": f1_score(y_test, predicted_labels, zero_division=0),
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_test, predicted_scores) if predicted_scores is not None else None,
        "pr_auc": average_precision_score(y_test, predicted_scores) if predicted_scores is not None else None,
    })

evaluation_report = pd.DataFrame(all_metrics)
evaluation_report.to_csv("model_evaluation_report.csv", index=False)
print(evaluation_report)

comparable = evaluation_report[evaluation_report["model_name"] != "baseline_most_frequent"]
best_row = comparable.sort_values(by="roc_auc", ascending=False).iloc[0]
best_model_name = best_row["model_name"]
best_pipeline = fitted_pipelines[best_model_name]
print("\nBest model:", best_model_name)

joblib.dump(best_pipeline, "trained_bioai_model.joblib")

metadata = {
    "model_name": best_model_name,
    "target_column": TARGET_COLUMN,
    "input_features": INPUT_FEATURES,
    "data_source": "ml_model_features.csv (only dataset used)",
    "target_definition": (
        "1 = positive association example in the dataset's construction, "
        "0 = negative example. A label of 0 does not necessarily mean the "
        "drug is clinically ineffective."
    ),
    "selected_model_metrics": best_row.to_dict(),
    "limitations": [
        "Trained on one dataset and one split; may not generalize.",
        "A high score means statistical similarity to positive training examples, not proof of clinical effectiveness.",
    ],
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("\nComputing permutation importance...")
result = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1)
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_importance_roc_auc_drop": result.importances_mean,
    "std_importance": result.importances_std,
}).sort_values(by="mean_importance_roc_auc_drop", ascending=False)
importance_df.to_csv("explainability_permutation_importance.csv", index=False)
print(importance_df)

print("\nDone. trained_bioai_model.joblib regenerated with your local scikit-learn version.")