"""
model_training.py
------------------
This is part of LAYER 2 of the BIO AI project: training and evaluating a
machine learning model that predicts Association_Label from a
Disease + Gene + Variant + Drug combination.

What "Association_Label" actually means in this dataset:
    - 1 = this Disease-Gene-Variant-Drug combination was treated as a
      POSITIVE example during dataset construction.
    - 0 = this combination was treated as a NEGATIVE example.

    Important: a label of 0 does NOT mean "this drug does not work" or
    "this gene-drug pair is biologically unrelated." The data-quality
    investigation (see data_processing.py / data_quality_report.csv)
    found that:
      - Every Gene+Variant group (Group_ID) appears in only one of the
        train/test splits (no leakage), and
      - The positive label rate is a flat 25% in both splits.
    This flat, identical rate strongly suggests the negative (label 0)
    examples were generated as part of building a balanced-ish machine
    learning dataset, rather than each one being an individually
    confirmed "this drug has no effect" biological finding. This model
    therefore predicts "does this combination look similar to the
    positive examples in this dataset", not "is this drug clinically
    effective".

What this file does:
    1. Builds a preprocessing + model pipeline (so preprocessing and the
       model are always applied together, during both training and
       prediction).
    2. Trains several candidate models: a DummyClassifier baseline,
       Logistic Regression, Random Forest, and Gradient Boosting.
    3. Evaluates every model on a held-out test set using multiple
       metrics (not just accuracy).
    4. Picks the best model based on validation performance.
    5. Saves the trained pipeline (preprocessing + model together) to a
       .joblib file, so that predictions later use the exact same
       preprocessing as training.
    6. Saves an evaluation report and model metadata.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# The four biological columns used as model inputs.
# We deliberately do NOT use Row_ID, Group_ID, or Split as predictors:
# these are dataset bookkeeping columns, not biological information, and
# using them could leak information or create a model that only works on
# this exact dataset.
INPUT_FEATURES = ["Disease", "Gene", "Variant", "Drug"]
TARGET_COLUMN = "Association_Label"

RANDOM_SEED = 42


def build_preprocessing_and_model_pipeline(model):
    """
    Build a scikit-learn Pipeline that first preprocesses the four
    biological input columns and then feeds them into a classifier.

    Why we need this:
        Disease, Gene, Variant, and Drug are all text categories (for
        example "Breast Cancer", "CYP2D6"). Most machine learning models
        need numbers, not text. OneHotEncoder converts each category into
        a set of 0/1 columns (for example, a column for "is this row
        Breast Cancer?").

        Wrapping preprocessing and the model together in one Pipeline
        means:
          - We only ever fit the encoder on the TRAINING data, never on
            test data (this prevents a subtle form of data leakage).
          - At prediction time, new data automatically goes through the
            exact same preprocessing steps as the training data.
          - A brand-new category not seen during training (for example,
            a drug name that never appeared in training) is handled
            safely instead of crashing, because of
            `handle_unknown="ignore"`.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "encode_categories",
                OneHotEncoder(handle_unknown="ignore"),
                INPUT_FEATURES,
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
    return pipeline


def get_candidate_models():
    """
    Return a dictionary of the candidate models we will train and
    compare. Each one is explained briefly below in beginner-friendly
    terms.
    """

    candidate_models = {
        # A baseline that ignores the input features and always predicts
        # the most common label. If our real models cannot beat this,
        # they are not actually learning anything useful.
        "baseline_most_frequent": DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_SEED
        ),
        # A simple, interpretable linear model. Easy to explain: each
        # category gets a weight, and the weights are added up to decide
        # the prediction.
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        # An ensemble of many decision trees. Usually more accurate than
        # a single linear model, and still reasonably easy to explain.
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
        ),
        # Similar to Random Forest (many decision trees voting together),
        # but each tree is built using more randomness when choosing how
        # to split the data. This often trains faster and can sometimes
        # generalize better than Random Forest.
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
        ),
        # Builds trees one after another, where each new tree tries to
        # correct the mistakes of the previous ones. Often very accurate,
        # at the cost of being a little harder to explain than Random
        # Forest.
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }
    return candidate_models


def split_data_using_existing_split_column(data):
    """
    Split the dataset into training and test sets using the dataset's own
    'Split' column, instead of randomly splitting it ourselves.

    Why: the data-quality check in data_processing.py already confirmed
    that this existing Split column respects Group_ID (Gene+Variant)
    boundaries, so every Gene+Variant group appears in only ONE of the
    two splits. This is exactly the property a group-aware split is
    meant to guarantee, so re-splitting randomly would only add risk of
    leakage without any benefit.
    """

    if "Split" not in data.columns:
        raise ValueError(
            "The dataset does not contain a 'Split' column, so the "
            "existing train/test split cannot be used."
        )

    train_data = data[data["Split"] == "train"].copy()
    test_data = data[data["Split"] == "test"].copy()

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError(
            "The 'Split' column did not produce both a non-empty train "
            "set and a non-empty test set."
        )

    return train_data, test_data


def compute_evaluation_metrics(model_name, true_labels, predicted_labels, predicted_scores):
    """
    Compute a set of standard classification metrics for one model's
    predictions on the test set, and return them as a dictionary.

    predicted_scores should be the model's predicted probability of the
    positive class (label = 1), used for ROC-AUC and PR-AUC. Some models
    (like a basic DummyClassifier) can still produce these via
    predict_proba.
    """

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()

    # Specificity = true negative rate = how well the model recognizes
    # negative examples. scikit-learn does not have a one-line function
    # for this, so we compute it directly from the confusion matrix.
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
        "specificity": specificity,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # ROC-AUC and PR-AUC need predicted scores (probabilities), not just
    # the final 0/1 predictions. If a model could not produce scores for
    # some reason, we record that honestly instead of guessing a number.
    if predicted_scores is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(true_labels, predicted_scores)
            metrics["pr_auc_average_precision"] = average_precision_score(
                true_labels, predicted_scores
            )
        except ValueError as error:
            metrics["roc_auc"] = None
            metrics["pr_auc_average_precision"] = None
            metrics["score_based_metric_error"] = str(error)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc_average_precision"] = None

    return metrics


def train_and_evaluate_all_models(data):
    """
    Train every candidate model on the training split, evaluate every one
    of them on the test split, and return:
        - a dictionary of {model_name: fitted pipeline}
        - a pandas DataFrame comparing their evaluation metrics
    """

    train_data, test_data = split_data_using_existing_split_column(data)

    X_train = train_data[INPUT_FEATURES]
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data[INPUT_FEATURES]
    y_test = test_data[TARGET_COLUMN]

    candidate_models = get_candidate_models()
    fitted_pipelines = {}
    all_metrics = []

    for model_name, model in candidate_models.items():
        print(f"Training model: {model_name} ...")
        pipeline = build_preprocessing_and_model_pipeline(model)
        pipeline.fit(X_train, y_train)

        predicted_labels = pipeline.predict(X_test)

        # Get predicted probability of the positive class, if available.
        predicted_scores = None
        if hasattr(pipeline, "predict_proba"):
            predicted_scores = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_evaluation_metrics(
            model_name, y_test, predicted_labels, predicted_scores
        )
        all_metrics.append(metrics)
        fitted_pipelines[model_name] = pipeline

    evaluation_report = pd.DataFrame(all_metrics)
    return fitted_pipelines, evaluation_report, (X_train, y_train, X_test, y_test)


def select_best_model(evaluation_report, fitted_pipelines, metric="roc_auc"):
    """
    Pick the best-performing model based on a chosen metric (ROC-AUC by
    default, since it is a reasonable general-purpose ranking metric that
    does not depend on choosing a probability threshold).

    We deliberately exclude the baseline model from being selected as the
    "best" model — it exists only for comparison, to make sure the real
    models are actually learning something.
    """

    comparable_rows = evaluation_report[
        evaluation_report["model_name"] != "baseline_most_frequent"
    ]
    best_row = comparable_rows.sort_values(by=metric, ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    best_pipeline = fitted_pipelines[best_model_name]

    return best_model_name, best_pipeline


def save_model_and_metadata(
    pipeline,
    model_name,
    evaluation_report,
    output_model_path="models/trained_bioai_model.joblib",
    output_metadata_path="models/model_metadata.json",
):
    """
    Save the trained pipeline (preprocessing + model together) to disk,
    along with a metadata file describing how it was built. Saving the
    metadata alongside the model is important so that anyone reusing the
    model later (including a future Streamlit app) understands exactly
    what it does and does not represent.
    """

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(pipeline, output_model_path)

    best_row = evaluation_report[evaluation_report["model_name"] == model_name].iloc[0]

    metadata = {
        "project_name": "BIO AI: Evidence-Guided Disease-Gene-Variant-Drug "
        "Association Retrieval and ML-Based Candidate Ranking System",
        "model_name": model_name,
        "target_column": TARGET_COLUMN,
        "target_definition": (
            "Association_Label: 1 = combination was a positive example in "
            "the source dataset's construction, 0 = negative example. "
            "A label of 0 does not necessarily mean the drug is clinically "
            "ineffective; see the README and data_quality_report.csv for "
            "the full investigation of what the negative examples "
            "represent."
        ),
        "input_features": INPUT_FEATURES,
        "data_source": "ml_model_features.csv (the only dataset used)",
        "training_approach": (
            "scikit-learn Pipeline combining a OneHotEncoder (for the four "
            "categorical biological features) with a classifier. Trained "
            "and evaluated using the dataset's own pre-existing Split "
            "column, which was verified beforehand to keep every "
            "Gene+Variant group entirely within one split (no group "
            "leakage between train and test)."
        ),
        "evaluation_method": (
            "Hold-out evaluation on the dataset's existing test split. "
            "Metrics: accuracy, precision, recall, F1-score, specificity, "
            "ROC-AUC, PR-AUC (average precision), and confusion matrix. "
            "Compared against a DummyClassifier baseline."
        ),
        "selected_model_metrics": best_row.to_dict(),
        "limitations": [
            "This model is trained on a single dataset and a single "
            "train/test split; its performance may not generalize to "
            "other datasets or real-world clinical settings.",
            "A high prediction score means the input combination looks "
            "statistically similar to the dataset's positive examples. "
            "It is NOT proof of clinical effectiveness, drug safety, or "
            "FDA approval.",
            "The model has only ever seen Disease, Gene, Variant, and "
            "Drug categories that exist in this dataset. Genuinely new "
            "categories are handled through OneHotEncoder's "
            "'unknown category' mechanism, which effectively treats them "
            "as having no specific prior association.",
            "Feature importance or model coefficients (see explainability "
            "outputs) describe statistical patterns learned from this "
            "dataset, not confirmed biological causality.",
        ],
        "random_seed": RANDOM_SEED,
    }

    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    with open(output_metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)

    return output_model_path, output_metadata_path


def save_evaluation_report(
    evaluation_report, output_path="outputs/model_evaluation_report.csv"
):
    """
    Save the model comparison table (one row per candidate model, with its
    metrics) to a CSV file. This is one of the required project
    deliverables.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    evaluation_report.to_csv(output_path, index=False)
    return output_path


def explain_model_with_permutation_importance(
    pipeline, X_test, y_test, n_repeats=10
):
    """
    Explainability for Layer 2: permutation feature importance.

    How this works, in simple terms:
        For each input feature (Disease, Gene, Variant, Drug) in turn,
        this function randomly shuffles that one column in the test data
        and checks how much worse the model's predictions become. If
        shuffling a feature hurts performance a lot, the model was
        relying on that feature heavily. If shuffling it barely changes
        anything, the model was not relying on it much.

    What this DOES tell us:
        Which input fields the trained model leans on most when making
        predictions on this dataset.

    What this does NOT tell us:
        It does not prove that a feature biologically CAUSES the
        association label. It only describes a statistical pattern that
        the model learned from this particular dataset. Correlation
        picked up by a model is not the same thing as a confirmed
        biological mechanism.

    Returns
    -------
    importance_df : pandas DataFrame
        One row per input feature, sorted from most to least important,
        with the average performance drop and its variability.
    """

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_importance_roc_auc_drop": result.importances_mean,
            "std_importance": result.importances_std,
        }
    ).sort_values(by="mean_importance_roc_auc_drop", ascending=False)

    return importance_df


def load_trained_model(model_path="models/trained_bioai_model.joblib"):
    """
    Load a previously trained and saved pipeline (preprocessing + model
    together) from disk, ready to use for predictions.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model file found at '{model_path}'. "
            f"Run train_model.py first to train and save a model."
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    # Quick manual test: train and evaluate all models, and print the
    # comparison table. Run with: python3 src/model_training.py
    from data_processing import load_dataset

    print("Loading dataset...")
    dataset = load_dataset("data/ml_model_features.csv")

    print("\nTraining and evaluating all candidate models...")
    pipelines, report, split_data = train_and_evaluate_all_models(dataset)

    print("\nEvaluation report:")
    print(report.to_string(index=False))

    best_name, best_pipeline = select_best_model(report, pipelines)
    print(f"\nBest model selected (by ROC-AUC, excluding baseline): {best_name}")

    model_path, metadata_path = save_model_and_metadata(best_pipeline, best_name, report)
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")

    report_path = save_evaluation_report(report)
    print(f"Saved evaluation report to: {report_path}")

    print("\nComputing permutation feature importance for the best model...")
    train_data, test_data = split_data_using_existing_split_column(dataset)
    X_test = test_data[INPUT_FEATURES]
    y_test = test_data[TARGET_COLUMN]
    importance_df = explain_model_with_permutation_importance(best_pipeline, X_test, y_test)
    print(importance_df.to_string(index=False))
