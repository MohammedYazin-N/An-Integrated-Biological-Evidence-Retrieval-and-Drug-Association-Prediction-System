"""
data_processing.py
===================

This file loads the BIO AI dataset (ml_model_features.csv), checks its
quality, and creates exploratory charts. It is written to be read top to
bottom like a lab notebook: each function does one clear job.

What this file does NOT do:
    - It does not fabricate biological facts. Everything it prints or plots
      comes directly from counting rows in the dataset.
    - It does not decide that Association_Label = 0 means "the drug does
      not work". See the long comment above `label_quality_report()` for
      why that assumption would be wrong for this dataset.

Functions a future Streamlit app (or any other script) can import:
    load_dataset(path)          -> pandas.DataFrame (the raw, untouched data)
    get_dataset_summary(df)     -> dict of simple counts, safe to display
    run_data_quality_checks(df) -> dict with every check described in the
                                    project brief, plus a one-row-per-check
                                    DataFrame you can save as a CSV report
    run_eda(df, output_dir)     -> saves PNG charts to output_dir and
                                    returns the list of file paths created
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; we only save PNG files
import matplotlib.pyplot as plt
import seaborn as sns

# Columns we always expect. We still check for them instead of assuming,
# because the project brief asks us to "verify the actual schema".
EXPECTED_COLUMNS = [
    "Row_ID", "Group_ID", "Split", "Disease", "Gene", "Variant", "Drug",
    "Variant_Type", "N_Alleles", "Drug_Is_Combination", "Association_Label",
]

# The four biological columns every prediction and search is built on.
BIOLOGICAL_COLUMNS = ["Disease", "Gene", "Variant", "Drug"]

TARGET_COLUMN = "Association_Label"


def load_dataset(path="data/ml_model_features.csv"):
    """
    Load ml_model_features.csv from disk and return it unchanged.

    We never edit this returned DataFrame in place anywhere else in the
    project. Every other function either reads from it or makes its own
    copy first (df.copy()), so this "original" DataFrame always stays a
    faithful copy of the CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find the dataset at '{path}'. "
            "This project only works with ml_model_features.csv — "
            "make sure the file is in the data/ folder."
        )
    df = pd.read_csv(path)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The dataset is missing expected column(s): {missing_cols}. "
            "The rest of this project assumes these columns exist."
        )
    return df


def get_dataset_summary(df):
    """
    Return a small dictionary of dataset-level facts: how many rows,
    how many unique diseases/genes/variants/drugs, and the label split.

    This is meant to be shown directly to a user (e.g. in a future
    Streamlit sidebar), so every value here is a plain number or string.
    """
    summary = {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "unique_diseases": df["Disease"].nunique(),
        "unique_genes": df["Gene"].nunique(),
        "unique_variants": df["Variant"].nunique(),
        "unique_drugs": df["Drug"].nunique(),
        "unique_gene_variant_groups": df["Group_ID"].nunique(),
        "positive_label_count": int((df[TARGET_COLUMN] == 1).sum()),
        "negative_label_count": int((df[TARGET_COLUMN] == 0).sum()),
        "positive_label_fraction": round(float((df[TARGET_COLUMN] == 1).mean()), 4),
    }
    if "Split" in df.columns:
        summary["train_rows"] = int((df["Split"] == "train").sum())
        summary["test_rows"] = int((df["Split"] == "test").sum())
    return summary


def run_data_quality_checks(df):
    """
    Run every data-quality check the project brief asks for and return:
        - a dictionary of results (for printing / using in code)
        - a tidy DataFrame (one row per check) you can save as a CSV report

    IMPORTANT — reading the Association_Label column correctly:
    This dataset's label is NOT "does this drug treat this disease
    biologically?". It is better read as "was this exact
    (Disease, Gene, Variant, Drug) row present as a confirmed, curated
    interaction (label=1), or was it constructed as a negative training
    example (label=0)?" We cannot tell from this file alone how every
    negative example was generated (that detail lived in an earlier
    dataset this project is intentionally not using). So this function
    treats label=0 as "not a confirmed positive example in this dataset",
    never as "proven not to work".
    """
    results = {}

    # 1. Shape
    results["n_rows"] = len(df)
    results["n_columns"] = df.shape[1]

    # 2. Data types
    results["dtypes"] = df.dtypes.astype(str).to_dict()

    # 3. Missing values
    missing_counts = df.isna().sum()
    results["missing_values_per_column"] = missing_counts[missing_counts > 0].to_dict()
    results["total_missing_values"] = int(missing_counts.sum())

    # 4. Duplicate full rows (ignoring the Row_ID, which is unique by design)
    non_id_cols = [c for c in df.columns if c != "Row_ID"]
    results["duplicate_rows_excluding_row_id"] = int(df.duplicated(subset=non_id_cols).sum())

    # 5. Duplicate biological combinations (Disease, Gene, Variant, Drug)
    results["duplicate_bio_combinations"] = int(df.duplicated(subset=BIOLOGICAL_COLUMNS).sum())

    # 6. Unique counts
    for col in BIOLOGICAL_COLUMNS:
        results[f"unique_{col.lower()}"] = int(df[col].nunique())

    # 7. Label distribution
    label_counts = df[TARGET_COLUMN].value_counts().to_dict()
    results["label_counts"] = label_counts
    results["label_positive_fraction"] = round(float(df[TARGET_COLUMN].mean()), 4)

    # 8. Conflicting labels: does the same (Disease, Gene, Variant, Drug)
    #    combination ever appear with BOTH label 0 and label 1?
    label_nunique_per_combo = df.groupby(BIOLOGICAL_COLUMNS)[TARGET_COLUMN].nunique()
    conflicting_combos = int((label_nunique_per_combo > 1).sum())
    results["conflicting_label_combinations"] = conflicting_combos

    # 9. Split column inspection
    if "Split" in df.columns:
        results["split_counts"] = df["Split"].value_counts().to_dict()
        results["label_rate_by_split"] = df.groupby("Split")[TARGET_COLUMN].mean().round(4).to_dict()
    else:
        results["split_counts"] = None

    # 10. Group_ID inspection and leakage risk check.
    #     Group_ID is "Gene || Variant". If the same Group_ID shows up in
    #     BOTH the train split and the test split, a model could partly
    #     "memorize" that gene-variant pair during training and look
    #     artificially good on the matching test rows. This is called
    #     group leakage.
    if "Group_ID" in df.columns and "Split" in df.columns:
        groups_per_split_count = df.groupby("Group_ID")["Split"].nunique()
        leaking_groups = int((groups_per_split_count > 1).sum())
        results["n_unique_groups"] = int(df["Group_ID"].nunique())
        results["groups_leaking_across_train_and_test"] = leaking_groups
        results["group_leakage_risk"] = (
            "NONE DETECTED — every Group_ID (Gene+Variant pair) appears in only one split."
            if leaking_groups == 0
            else f"WARNING — {leaking_groups} Group_ID value(s) appear in BOTH train and test."
        )
    else:
        results["group_leakage_risk"] = "Could not check: Group_ID or Split column missing."

    # 11. Known data-quality issue found during manual inspection:
    #     a handful of Variant values are actually gene FULL NAMES, not
    #     variant identifiers (e.g. "Solute Carrier Family 26 Member 9"
    #     instead of a real rsID/star-allele for that row).
    #     This is different from the legitimate "GSTM1 non-null; GSTM1 null"
    #     style entries, which ARE a real variant notation (zygosity/copy
    #     number) used in pharmacogenomics, not an error — so we must not
    #     flag those. We only flag Variant values that do not contain
    #     "null" and are not a recognized rsID/star-allele/HLA format.
    #     We flag these rather than silently changing or dropping them,
    #     per the project rule: "do not automatically delete records or
    #     resolve conflicts without documenting the rule used."
    looks_like_gene_full_name = (
        df["Variant"].str.contains(" ", na=False)
        & ~df["Variant"].str.contains("null", case=False, na=False)
        & (df["Variant_Type"] == "other")
    )
    results["rows_with_gene_name_in_variant_column"] = int(looks_like_gene_full_name.sum())
    results["rows_with_gene_name_in_variant_column_examples"] = (
        df.loc[looks_like_gene_full_name, ["Row_ID", "Gene", "Variant"]].to_dict("records")
    )

    # Separately: count the legitimate zygosity/copy-number style entries,
    # just so they are documented too (not an error, just worth knowing).
    zygosity_notation_mask = df["Variant"].str.contains("null", case=False, na=False)
    results["rows_with_zygosity_copy_number_notation"] = int(zygosity_notation_mask.sum())

    # Build the tidy, CSV-friendly report (one row per simple check).
    # Complex nested values (dicts/lists) are converted to strings so the
    # CSV stays readable in Excel.
    report_rows = []
    for key, value in results.items():
        report_rows.append({"check": key, "result": str(value)})
    report_df = pd.DataFrame(report_rows)

    return results, report_df


def run_eda(df, output_dir="outputs"):
    """
    Create and save a small set of exploratory charts as PNG files.
    Returns the list of file paths created.

    Every chart here is a plain count of what is IN the dataset. A gene
    appearing often in this file means it was curated often — it does not
    mean that gene is more biologically important than a rarer one.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    sns.set_theme(style="whitegrid")

    # --- Chart 1: Association_Label distribution ---
    fig, ax = plt.subplots(figsize=(5, 4))
    label_counts = df[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, ax=ax, color="#1B4B43")
    ax.set_xlabel("Association_Label (0 = not a confirmed positive example, 1 = positive example)")
    ax.set_ylabel("Number of rows")
    ax.set_title("How many rows have each label?")
    for i, v in enumerate(label_counts.values):
        ax.text(i, v + 20, str(v), ha="center")
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_label_distribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    # --- Chart 2: Unique counts of the four biological columns ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = {col: df[col].nunique() for col in BIOLOGICAL_COLUMNS}
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=ax, color="#2F6B60")
    ax.set_ylabel("Number of unique values")
    ax.set_title("Unique diseases, genes, variants, and drugs in the dataset")
    for i, v in enumerate(counts.values()):
        ax.text(i, v + 2, str(v), ha="center")
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_unique_counts.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    # --- Chart 3: Most common diseases ---
    fig, ax = plt.subplots(figsize=(7, 6))
    top_diseases = df["Disease"].value_counts().head(15)
    sns.barplot(x=top_diseases.values, y=top_diseases.index, ax=ax, color="#C9973E")
    ax.set_xlabel("Number of rows in the dataset")
    ax.set_title("15 most frequently curated diseases")
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_top_diseases.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    # --- Chart 4: Most common genes ---
    fig, ax = plt.subplots(figsize=(7, 6))
    top_genes = df["Gene"].value_counts().head(15)
    sns.barplot(x=top_genes.values, y=top_genes.index, ax=ax, color="#C9973E")
    ax.set_xlabel("Number of rows in the dataset")
    ax.set_title("15 most frequently curated genes")
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_top_genes.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    # --- Chart 5: Most common drugs ---
    fig, ax = plt.subplots(figsize=(7, 6))
    top_drugs = df["Drug"].value_counts().head(15)
    sns.barplot(x=top_drugs.values, y=top_drugs.index, ax=ax, color="#C9973E")
    ax.set_xlabel("Number of rows in the dataset")
    ax.set_title("15 most frequently curated drugs")
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_top_drugs.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    # --- Chart 6: Label rate for the 10 most common diseases ---
    fig, ax = plt.subplots(figsize=(8, 5))
    top10_diseases = df["Disease"].value_counts().head(10).index
    subset = df[df["Disease"].isin(top10_diseases)]
    label_rate = subset.groupby("Disease")[TARGET_COLUMN].mean().sort_values(ascending=False)
    sns.barplot(x=label_rate.values, y=label_rate.index, ax=ax, color="#1B4B43")
    ax.set_xlabel("Fraction of rows labeled as a positive association (label=1)")
    ax.set_title("Positive-label rate among the 10 most common diseases")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_label_rate_by_disease.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    # --- Chart 7: Variant_Type distribution ---
    fig, ax = plt.subplots(figsize=(5, 4))
    vt_counts = df["Variant_Type"].value_counts()
    sns.barplot(x=vt_counts.index, y=vt_counts.values, ax=ax, color="#2F6B60")
    ax.set_ylabel("Number of rows")
    ax.set_title("Variant type distribution")
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_variant_type_distribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    # --- Chart 8: Missing-value heatmap (will be empty if no missing values,
    #     which is itself a useful thing to show) ---
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax, cmap="Reds")
    ax.set_title(f"Missing-value map ({int(df.isna().sum().sum())} missing cells total)")
    fig.tight_layout()
    path = os.path.join(output_dir, "eda_missing_values.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved_files.append(path)

    return saved_files


if __name__ == "__main__":
    # Quick manual test: run this file directly to see a summary printed.
    data = load_dataset("data/ml_model_features.csv")
    print("Dataset summary:")
    for k, v in get_dataset_summary(data).items():
        print(f"  {k}: {v}")

    print("\nRunning data quality checks...")
    results, report_df = run_data_quality_checks(data)
    print(report_df.to_string(index=False))

    # Save the data quality report as a required project deliverable.
    os.makedirs("outputs", exist_ok=True)
    report_path = os.path.join("outputs", "data_quality_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved data quality report to {report_path}")

    print("\nSaving EDA charts to outputs/ ...")
    files = run_eda(data, output_dir="outputs")
    for f in files:
        print(f"  saved {f}")
