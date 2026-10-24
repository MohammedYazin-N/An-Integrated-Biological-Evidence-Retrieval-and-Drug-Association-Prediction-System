"""
prediction.py
--------------
This is part of LAYER 2 of the BIO AI project: using the trained model to
predict the Association_Label for a NEW Disease-Gene-Variant-Drug
combination (one that is not necessarily already in the dataset).

This file does NOT train anything. It only loads an already-trained model
(saved by model_training.py / train_model.py) and uses it to make
predictions.

Important honesty reminder (repeated here on purpose):
    A prediction from this model is a statistical estimate of how similar
    a new combination looks to the POSITIVE examples in the training
    dataset. It is NOT a clinical judgement, NOT a guarantee of drug
    effectiveness, and NOT an FDA approval status. Always treat the
    output as a starting point for further investigation, never as a
    final answer.
"""

import pandas as pd

# This try/except lets the file be imported two different ways:
#   - as part of the "src" package (e.g. "from src.prediction import ...",
#     used by train_model.py and a future Streamlit app at the project
#     root), or
#   - directly as a standalone script (e.g. "python3 src/prediction.py",
#     used for quick manual testing).
try:
    from src.model_training import load_trained_model, INPUT_FEATURES
except ImportError:
    from model_training import load_trained_model, INPUT_FEATURES


def validate_candidate_input(disease, gene, variant, drug):
    """
    Check that all four required fields were actually provided (not
    empty or missing) before trying to make a prediction.

    Returns a list of warning strings. An empty list means the input
    looks valid.
    """

    warnings = []
    fields = {"Disease": disease, "Gene": gene, "Variant": variant, "Drug": drug}

    for field_name, value in fields.items():
        if value is None or str(value).strip() == "":
            warnings.append(f"Missing required field: {field_name}.")

    return warnings


def flag_unfamiliar_categories(candidate_row, training_categories):
    """
    Check whether the Disease, Gene, Variant, or Drug values in a
    candidate row were ever seen during model training.

    Why this matters: the model's OneHotEncoder was fit only on
    categories that existed in the training data. A brand-new category
    (for example, a drug name the model has never seen) is still handled
    safely (see model_training.py's `handle_unknown="ignore"`), but the
    model effectively has NO learned information about it. This function
    produces a plain-language warning so nobody mistakes "the code ran
    without crashing" for "the model has real knowledge about this
    input."

    Parameters
    ----------
    candidate_row : dict
        A dictionary with keys Disease, Gene, Variant, Drug.
    training_categories : dict
        A dictionary mapping each of those same keys to the set of
        category values that were present in the training data.

    Returns
    -------
    warnings : list of str
    """

    warnings = []
    for field_name in INPUT_FEATURES:
        value = candidate_row.get(field_name)
        known_values = training_categories.get(field_name, set())
        if value is not None and value not in known_values:
            warnings.append(
                f"'{value}' was not seen in the training data for "
                f"'{field_name}'. The model has no specific learned "
                f"information about this value."
            )
    return warnings


def get_training_categories(training_data):
    """
    Build a dictionary of {column_name: set of unique values seen in
    training}, used by flag_unfamiliar_categories() to detect brand-new
    categories at prediction time.
    """

    training_categories = {}
    for field_name in INPUT_FEATURES:
        training_categories[field_name] = set(training_data[field_name].unique())
    return training_categories


def predict_association(
    disease, gene, variant, drug, pipeline, training_categories=None
):
    """
    Predict the Association_Label for one Disease-Gene-Variant-Drug
    combination.

    Parameters
    ----------
    disease, gene, variant, drug : str
        The candidate combination to predict on.
    pipeline : a fitted scikit-learn Pipeline
        The trained preprocessing + model pipeline (load with
        load_trained_model()).
    training_categories : dict or None
        Optional. If given (from get_training_categories()), unfamiliar
        category warnings will be included in the result.

    Returns
    -------
    result : dict
        A plain dictionary describing the prediction, meant to be easy
        to use from another Python script or a future Streamlit app.
    """

    input_warnings = validate_candidate_input(disease, gene, variant, drug)
    if input_warnings:
        return {
            "Disease": disease,
            "Gene": gene,
            "Variant": variant,
            "Drug": drug,
            "Predicted_Label": None,
            "Predicted_Association_Score": None,
            "Warnings": input_warnings,
        }

    candidate_row = {"Disease": disease, "Gene": gene, "Variant": variant, "Drug": drug}
    candidate_df = pd.DataFrame([candidate_row])[INPUT_FEATURES]

    predicted_label = int(pipeline.predict(candidate_df)[0])

    predicted_score = None
    if hasattr(pipeline, "predict_proba"):
        # Column 1 is the predicted probability of the positive class
        # (Association_Label == 1), because the model was trained with
        # 0/1 labels in that order.
        predicted_score = float(pipeline.predict_proba(candidate_df)[0, 1])

    warnings = []
    if training_categories is not None:
        warnings = flag_unfamiliar_categories(candidate_row, training_categories)

    return {
        "Disease": disease,
        "Gene": gene,
        "Variant": variant,
        "Drug": drug,
        "Predicted_Label": predicted_label,
        "Predicted_Association_Score": predicted_score,
        "Warnings": warnings,
    }


if __name__ == "__main__":
    # Quick manual test. Run from the project root with:
    #     python3 src/prediction.py
    try:
        from src.data_processing import load_dataset
    except ImportError:
        from data_processing import load_dataset

    print("Loading dataset and trained model for a demonstration...")
    dataset = load_dataset("data/ml_model_features.csv")
    pipeline = load_trained_model("models/trained_bioai_model.joblib")
    training_categories = get_training_categories(dataset)

    example = dataset.iloc[0]
    result = predict_association(
        disease=example["Disease"],
        gene=example["Gene"],
        variant=example["Variant"],
        drug=example["Drug"],
        pipeline=pipeline,
        training_categories=training_categories,
    )
    print("\nExample prediction (using a real row from the dataset):")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nExample prediction with a made-up, unfamiliar drug name:")
    result2 = predict_association(
        disease=example["Disease"],
        gene=example["Gene"],
        variant=example["Variant"],
        drug="SomeBrandNewDrugXYZ",
        pipeline=pipeline,
        training_categories=training_categories,
    )
    for key, value in result2.items():
        print(f"  {key}: {value}")
