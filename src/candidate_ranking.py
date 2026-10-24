"""
candidate_ranking.py
---------------------
This is part of LAYER 2 of the BIO AI project: ranking a LIST of
candidate Disease-Gene-Variant-Drug combinations by their predicted
association score, from most to least "positive-looking" according to
the trained model.

This file builds directly on prediction.py (which predicts one
combination at a time) and simply applies it to many candidates, then
sorts and formats the results.

Important honesty reminder:
    Ranking candidates by predicted score tells you which combinations
    the model considers MOST SIMILAR to the positive training examples.
    It does NOT identify the clinically most effective drug, and a
    top-ranked candidate is not a treatment recommendation.
"""

import os

import pandas as pd

try:
    from src.prediction import predict_association, validate_candidate_input
except ImportError:
    from prediction import predict_association, validate_candidate_input

REQUIRED_FIELDS = ["Disease", "Gene", "Variant", "Drug"]


def rank_candidate_combinations(candidates, pipeline, training_categories=None):
    """
    Score and rank a list of candidate combinations.

    Parameters
    ----------
    candidates : list of dict
        Each dictionary must have the keys "Disease", "Gene", "Variant",
        and "Drug". Example:
            [
                {"Disease": "Breast Cancer", "Gene": "CYP2D6",
                 "Variant": "rs3892097", "Drug": "Tamoxifen"},
                {"Disease": "Breast Cancer", "Gene": "CYP2D6",
                 "Variant": "rs3892097", "Drug": "Letrozole"},
            ]
    pipeline : a fitted scikit-learn Pipeline
        The trained preprocessing + model pipeline.
    training_categories : dict or None
        Optional, passed through to predict_association() so that
        unfamiliar-category warnings are included.

    Returns
    -------
    ranked_results : pandas DataFrame
        Columns: Disease, Gene, Variant, Drug, Predicted_Label,
        Predicted_Association_Score, Ranking, Warnings.
        Sorted from the highest predicted score to the lowest. Candidates
        with missing/invalid fields are still included, but are placed at
        the bottom (their score is treated as missing) and their
        Warnings column explains why.
    """

    if not isinstance(candidates, list) or len(candidates) == 0:
        raise ValueError(
            "candidates must be a non-empty list of dictionaries, each "
            "with Disease, Gene, Variant, and Drug keys."
        )

    scored_rows = []
    for candidate in candidates:
        missing_fields = [field for field in REQUIRED_FIELDS if field not in candidate]
        if missing_fields:
            scored_rows.append(
                {
                    "Disease": candidate.get("Disease"),
                    "Gene": candidate.get("Gene"),
                    "Variant": candidate.get("Variant"),
                    "Drug": candidate.get("Drug"),
                    "Predicted_Label": None,
                    "Predicted_Association_Score": None,
                    "Warnings": [f"Missing required key(s): {missing_fields}"],
                }
            )
            continue

        result = predict_association(
            disease=candidate["Disease"],
            gene=candidate["Gene"],
            variant=candidate["Variant"],
            drug=candidate["Drug"],
            pipeline=pipeline,
            training_categories=training_categories,
        )
        scored_rows.append(result)

    results_df = pd.DataFrame(scored_rows)

    # Sort so that candidates with a real score come first (highest
    # score first), and candidates with no score (because of invalid
    # input) are pushed to the bottom rather than crashing the sort.
    results_df = results_df.sort_values(
        by="Predicted_Association_Score", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # Ranking is just the row's position after sorting (1 = top-ranked).
    results_df["Ranking"] = results_df.index + 1

    # Put columns in the clear, documented output order from the
    # project specification.
    ordered_columns = [
        "Disease",
        "Gene",
        "Variant",
        "Drug",
        "Predicted_Label",
        "Predicted_Association_Score",
        "Ranking",
        "Warnings",
    ]
    results_df = results_df[ordered_columns]

    return results_df


def export_ranking_results(
    ranked_results, output_path="outputs/candidate_ranking_results.csv"
):
    """
    Save a candidate-ranking DataFrame (from rank_candidate_combinations)
    to a CSV file.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # The Warnings column holds Python lists, which do not save cleanly
    # to CSV by default. Convert each list to a readable, semicolon
    # separated string first.
    exportable = ranked_results.copy()
    exportable["Warnings"] = exportable["Warnings"].apply(
        lambda warning_list: "; ".join(warning_list) if warning_list else ""
    )

    exportable.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # Quick manual test. Run from the project root with:
    #     python3 src/candidate_ranking.py
    try:
        from src.data_processing import load_dataset
        from src.model_training import load_trained_model
        from src.prediction import get_training_categories
    except ImportError:
        from data_processing import load_dataset
        from model_training import load_trained_model
        from prediction import get_training_categories

    print("Loading dataset and trained model for a demonstration...")
    dataset = load_dataset("data/ml_model_features.csv")
    pipeline = load_trained_model("models/trained_bioai_model.joblib")
    training_categories = get_training_categories(dataset)

    # Build a small example list of candidates using real values that
    # exist in the dataset, so the demonstration is meaningful.
    example_rows = dataset.sample(5, random_state=1)
    candidates = example_rows[REQUIRED_FIELDS].to_dict(orient="records")

    print(f"\nRanking {len(candidates)} example candidates...")
    ranked = rank_candidate_combinations(candidates, pipeline, training_categories)
    print(ranked.to_string(index=False))

    output_file = export_ranking_results(ranked)
    print(f"\nSaved ranked results to: {output_file}")
