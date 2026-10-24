"""
evidence_retrieval.py
----------------------
This is LAYER 1 of the BIO AI project: the Biological Evidence Retrieval
Engine.

What this file does:
    It lets you search the dataset (ml_model_features.csv) for existing
    records that match a Disease, Gene, Variant, and/or Drug — using any
    ONE of these on its own, or any combination of them together.

What this file does NOT do:
    It does not predict anything and it does not use machine learning.
    It only looks up rows that are ALREADY in the dataset. If a search
    returns no rows, that only means "no matching row was found in this
    dataset" — it does NOT mean the biological relationship does not
    exist. Absence of a record is not evidence of absence.

Important honesty note:
    The dataset tells us whether a Disease-Gene-Variant-Drug combination
    was labelled as a positive or negative association example for
    machine learning purposes. It does NOT tell us the mechanism of
    action, the FDA approval status, or whether the relationship is
    "clinically established." This file never invents such information.
    It only reports what is actually present in the dataset columns.
"""

import os
import pandas as pd

# The biological columns that a person can search by.
SEARCHABLE_COLUMNS = ["Disease", "Gene", "Variant", "Drug"]


def search_biological_records(
    data,
    disease=None,
    gene=None,
    variant=None,
    drug=None,
    association_label=None,
    exact_match=False,
):
    """
    Search the dataset for rows matching any combination of Disease, Gene,
    Variant, and/or Drug.

    You do NOT need to provide all four. You can search with just one
    (for example, only a gene name) and this function will return every
    row where that gene appears, regardless of what the other columns are.

    Parameters
    ----------
    data : pandas DataFrame
        The dataset to search (normally the output of load_dataset()).
    disease, gene, variant, drug : str or None
        The search text for each column. Leave as None to not filter on
        that column at all.
    association_label : int or None
        If set to 0 or 1, only keep rows with that Association_Label.
        Leave as None to keep both positive and negative rows.
    exact_match : bool
        If True, a search term must match a cell exactly (after making
        both lowercase). If False (the default), a search term only has
        to appear somewhere inside the cell (partial / substring match).
        Both modes are case-insensitive.

    Returns
    -------
    matching_rows : pandas DataFrame
        All rows in `data` that matched every search term that was given.
        If nothing was matched, this DataFrame will have zero rows.
    number_of_matches : int
        A simple count of how many rows matched, for convenience.
    """

    # Start with a mask that marks every row as "still a candidate".
    # We will narrow this mask down for each search term that was given.
    still_matching = pd.Series(True, index=data.index)

    # Build the list of (column_name, search_term) pairs that were
    # actually provided, skipping any that were left as None.
    search_terms = {
        "Disease": disease,
        "Gene": gene,
        "Variant": variant,
        "Drug": drug,
    }

    for column_name, term in search_terms.items():
        if term is None or str(term).strip() == "":
            # The user did not ask to filter on this column, so skip it.
            continue

        if column_name not in data.columns:
            # The dataset does not have this column at all.
            raise ValueError(
                f"The dataset does not contain a '{column_name}' column, "
                f"so it is not possible to search by {column_name}."
            )

        column_values = data[column_name].astype(str).str.lower()
        search_text = str(term).strip().lower()

        if exact_match:
            column_matches = column_values == search_text
        else:
            column_matches = column_values.str.contains(
                search_text, regex=False, na=False
            )

        # A row only stays a match if it also matches this new condition.
        still_matching = still_matching & column_matches

    # Optionally filter by Association_Label (0 = negative, 1 = positive).
    if association_label is not None:
        if "Association_Label" not in data.columns:
            raise ValueError(
                "The dataset does not contain an 'Association_Label' column."
            )
        still_matching = still_matching & (
            data["Association_Label"] == association_label
        )

    matching_rows = data[still_matching].copy()
    number_of_matches = len(matching_rows)

    return matching_rows, number_of_matches


def summarize_search_results(matching_rows):
    """
    Given a DataFrame of matching rows (from search_biological_records),
    build a small dictionary that summarizes what is in them: how many
    unique diseases, genes, variants, and drugs appear, and how the
    Association_Label is split between positive and negative.

    This is useful for quickly describing a search result without having
    to read every row.
    """

    if len(matching_rows) == 0:
        return {
            "number_of_records": 0,
            "message": "No matching record found in the current dataset.",
        }

    summary = {"number_of_records": len(matching_rows)}

    for column_name in SEARCHABLE_COLUMNS:
        if column_name in matching_rows.columns:
            summary[f"unique_{column_name.lower()}s"] = int(
                matching_rows[column_name].nunique()
            )

    if "Association_Label" in matching_rows.columns:
        summary["positive_label_count"] = int(
            (matching_rows["Association_Label"] == 1).sum()
        )
        summary["negative_label_count"] = int(
            (matching_rows["Association_Label"] == 0).sum()
        )

    return summary


def get_unique_values(data, column_name):
    """
    Return a sorted list of every unique, non-missing value in a given
    biological column (for example, every unique Disease name).

    This is handy for building dropdown menus or for checking the exact
    spelling the dataset uses for a disease, gene, variant, or drug.
    """

    if column_name not in data.columns:
        raise ValueError(f"The dataset does not contain a '{column_name}' column.")

    unique_values = data[column_name].dropna().unique().tolist()
    unique_values.sort()
    return unique_values


def export_search_results(matching_rows, output_path):
    """
    Save a DataFrame of search results to a CSV file so it can be shared,
    inspected in Excel, or reused elsewhere.

    Parameters
    ----------
    matching_rows : pandas DataFrame
        The rows to save (normally the output of search_biological_records).
    output_path : str
        Where to save the CSV file, for example "outputs/my_search.csv".
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    matching_rows.to_csv(output_path, index=False)
    return output_path


def describe_evidence_type(matching_rows):
    """
    Produce a short, honest description of what kind of information a set
    of search results actually represents.

    This function exists specifically so that nobody (including a future
    Streamlit app) accidentally describes dataset rows as if they were
    independently verified clinical evidence. It always labels results as
    coming from the project's training/evidence dataset, not from an
    external clinical or regulatory source.
    """

    if len(matching_rows) == 0:
        return "No matching record found in the current dataset."

    n_positive = (
        int((matching_rows["Association_Label"] == 1).sum())
        if "Association_Label" in matching_rows.columns
        else None
    )
    n_negative = (
        int((matching_rows["Association_Label"] == 0).sum())
        if "Association_Label" in matching_rows.columns
        else None
    )

    message = (
        f"Found {len(matching_rows)} record(s) in the ml_model_features.csv "
        f"dataset. "
    )
    if n_positive is not None:
        message += (
            f"{n_positive} record(s) are labelled as a positive association "
            f"and {n_negative} record(s) are labelled as a negative "
            f"association in this dataset. "
        )
    message += (
        "These labels describe how the dataset was constructed for machine "
        "learning purposes. They are NOT a clinical approval status, a "
        "mechanism of action, or independently verified biological "
        "evidence. Always confirm clinically important claims against a "
        "primary scientific or regulatory source before relying on them."
    )
    return message


if __name__ == "__main__":
    # Quick manual test / demonstration of Layer 1.
    # This block only runs when you execute this file directly, e.g.:
    #     python3 src/evidence_retrieval.py
    from data_processing import load_dataset

    print("Loading dataset for a Layer 1 demonstration...")
    dataset = load_dataset("data/ml_model_features.csv")

    # Example 1: search using only a gene name.
    print("\nExample 1: search by Gene = 'CYP2D6' only")
    results, count = search_biological_records(dataset, gene="CYP2D6")
    print(f"  Found {count} matching record(s).")
    print(f"  {describe_evidence_type(results)}")
    print(f"  Summary: {summarize_search_results(results)}")

    # Example 2: search using a Disease + Drug combination.
    print("\nExample 2: search by Disease contains 'cancer' AND Drug = 'Tamoxifen'")
    results2, count2 = search_biological_records(
        dataset, disease="cancer", drug="Tamoxifen"
    )
    print(f"  Found {count2} matching record(s).")
    print(f"  {describe_evidence_type(results2)}")

    # Example 3: a search that should find nothing.
    print("\nExample 3: search by Gene = 'NOT_A_REAL_GENE'")
    results3, count3 = search_biological_records(dataset, gene="NOT_A_REAL_GENE")
    print(f"  {describe_evidence_type(results3)}")

    # Example 4: export a result to CSV.
    export_path = export_search_results(results, "outputs/example_search_results.csv")
    print(f"\nExample 1 results exported to: {export_path}")
