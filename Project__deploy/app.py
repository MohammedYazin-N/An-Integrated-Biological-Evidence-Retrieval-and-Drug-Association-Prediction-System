import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BIO AI", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# File paths (all files sit in the same folder as app.py)
# ----------------------------------------------------------------------
DATA_PATH = "ml_model_features.csv"
MODEL_PATH = "trained_bioai_model.joblib"
METADATA_PATH = "model_metadata.json"
EVALUATION_REPORT_PATH = "model_evaluation_report.csv"
IMPORTANCE_PATH = "explainability_permutation_importance.csv"

INPUT_FEATURES = ["Disease", "Gene", "Variant", "Drug"]
DISPLAY_COLUMNS = [
    "Disease", "Gene", "Variant", "Variant_Type",
    "Drug", "N_Alleles", "Drug_Is_Combination", "Association_Label",
]

TEAL_DARK = "#1B4B43"
TEAL = "#2F6B60"
ACCENT = "#4FB3A0"   # brighter teal, readable on a dark background
GOLD = "#E0B15A"
LABEL_COLORS = {"Positive": ACCENT, "Negative": GOLD}

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
    .hero {{
        background: linear-gradient(135deg, {TEAL_DARK} 0%, {TEAL} 100%);
        padding: 1.5rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;
    }}
    .hero h1 {{ color: #FFFFFF; margin: 0; font-size: 1.8rem; font-weight: 700; }}
    .hero p {{ color: #DCE9E5; margin: 0.35rem 0 0 0; font-size: 0.98rem; }}
    .card {{
        border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        padding: 1rem 1.3rem; background: rgba(128,128,128,0.06); height: 100%;
    }}
    .card.pos {{ border-left: 6px solid {ACCENT}; }}
    .card.neg {{ border-left: 6px solid {GOLD}; }}
    .card h3 {{ margin: 0.2rem 0 0.4rem 0; font-size: 1.25rem; }}
    .card .small {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7; }}
    div[data-testid="stMetric"] {{
        background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.2);
        padding: 0.8rem 1rem; border-radius: 12px;
    }}
    .stButton > button, .stFormSubmitButton > button {{ border-radius: 8px; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def disclaimer():
    st.caption(
        "Scientific note: model scores show statistical similarity to the training "
        "dataset's positive examples. They are not clinical advice, a measure of "
        "drug effectiveness, or an FDA approval status. A label of 0 in the data "
        "does not necessarily mean a drug is ineffective."
    )


# ----------------------------------------------------------------------
# Loading (cached, so nothing is reloaded on every click)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv_if_exists(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


@st.cache_data
def load_json_if_exists(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


data = load_data()
pipeline = load_model()

known_values = {
    "Disease": set(data["Disease"].str.lower()),
    "Gene": set(data["Gene"]),
    "Variant": set(data["Variant"]),
    "Drug": set(data["Drug"].str.lower()),
}
OPTIONS = {
    "Disease": sorted(data["Disease"].unique()),
    "Gene": sorted(data["Gene"].unique()),
    "Variant": sorted(data["Variant"].unique()),
    "Drug": sorted(data["Drug"].unique()),
}


def label_name(value):
    return "Positive" if int(value) == 1 else "Negative"


# ----------------------------------------------------------------------
# Layer 1: evidence search (free-text mode)
# ----------------------------------------------------------------------
def search_biological_records(df, disease=None, gene=None, variant=None, drug=None,
                              exact_match=False):
    mask = pd.Series(True, index=df.index)
    terms = {"Disease": disease, "Gene": gene, "Variant": variant, "Drug": drug}
    for col, term in terms.items():
        if term and term.strip():
            col_values = df[col].astype(str).str.lower()
            text = term.strip().lower()
            if exact_match:
                mask &= col_values == text
            else:
                mask &= col_values.str.contains(text, regex=False, na=False)
    return df[mask]


# ----------------------------------------------------------------------
# Layer 2: prediction
# ----------------------------------------------------------------------
def predict_association(disease, gene, variant, drug):
    values = {"Disease": disease, "Gene": gene, "Variant": variant, "Drug": drug}
    for name, value in values.items():
        if not value or not str(value).strip():
            return {"error": f"Please provide a value for {name}."}

    # Disease and Drug are stored in lowercase in this dataset.
    cleaned = {
        "Disease": disease.strip().lower(),
        "Gene": gene.strip(),
        "Variant": variant.strip(),
        "Drug": drug.strip().lower(),
    }
    row = pd.DataFrame([cleaned])[INPUT_FEATURES]
    label = int(pipeline.predict(row)[0])
    score = float(pipeline.predict_proba(row)[0, 1])

    warnings = [
        f"'{value}' was not seen in the training data for {col}, so the model has "
        f"no learned information about it."
        for col, value in cleaned.items()
        if value not in known_values[col]
    ]
    return {
        "Disease": disease, "Gene": gene, "Variant": variant, "Drug": drug,
        "Predicted_Label": label,
        "Predicted_Association_Score": score,
        "Warnings": warnings,
    }


def score_gauge(score, height=250):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            title={"text": "Association score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 0.5], "color": "#3B3222"},
                    {"range": [0.5, 1], "color": "#1F3F39"},
                ],
                "threshold": {"line": {"color": GOLD, "width": 4}, "value": 0.5},
            },
        )
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def show_prediction(result, key):
    if "error" in result:
        st.error(result["error"])
        return
    positive = result["Predicted_Label"] == 1
    headline = (
        "Resembles positive training examples"
        if positive else "Resembles negative training examples"
    )
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f'<div class="card {"pos" if positive else "neg"}">'
            f'<div class="small">Model output</div><h3>{headline}</h3>'
            f'Predicted label: <b>{label_name(result["Predicted_Label"])} '
            f'({result["Predicted_Label"]})</b><br>'
            f'Score: <b>{result["Predicted_Association_Score"]:.3f}</b></div>',
            unsafe_allow_html=True,
        )
        for w in result["Warnings"]:
            st.warning(w)
    with right:
        st.plotly_chart(
            score_gauge(result["Predicted_Association_Score"]),
            width="stretch", key=f"gauge_{key}",
        )


def example_picker(prefix):
    """Fill the four input boxes with a real row from the dataset."""
    row = data.sample(1).iloc[0]
    for col in INPUT_FEATURES:
        st.session_state[f"{prefix}_{col}"] = row[col]


def four_inputs(prefix):
    cols = st.columns(4)
    values = []
    for i, col in enumerate(INPUT_FEATURES):
        with cols[i]:
            values.append(
                st.selectbox(
                    col, OPTIONS[col], index=None, key=f"{prefix}_{col}",
                    placeholder=f"Select or type a {col.lower()}",
                    accept_new_options=True,
                )
            )
    return values


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
def page_overview():
    hero("BIO AI", "Evidence-guided disease-gene-variant-drug association retrieval "
                   "and ML-based candidate ranking")

    k = st.columns(6)
    k[0].metric("Records", f"{len(data):,}")
    k[1].metric("Diseases", data["Disease"].nunique())
    k[2].metric("Genes", data["Gene"].nunique())
    k[3].metric("Variants", data["Variant"].nunique())
    k[4].metric("Drugs", data["Drug"].nunique())
    k[5].metric("Positive labels", f"{data['Association_Label'].mean():.0%}")

    st.write("")
    c1, c2, c3 = st.columns([1, 1.3, 1.3])
    with c1:
        counts = data["Association_Label"].map(label_name).value_counts()
        fig = px.pie(names=counts.index, values=counts.values, hole=0.55,
                     color=counts.index, color_discrete_map=LABEL_COLORS,
                     title="Label distribution")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch", key="ov_pie")
    with c2:
        top = data["Disease"].value_counts().head(10).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     title="Most frequent diseases", labels={"x": "Records", "y": ""})
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch", key="ov_dis")
    with c3:
        top = data["Drug"].value_counts().head(10).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     title="Most frequent drugs", labels={"x": "Records", "y": ""})
        fig.update_traces(marker_color=GOLD)
        fig.update_xaxes(tickangle=0, nticks=4)
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch", key="ov_drug")

    st.caption("Frequency in the dataset reflects how the data was collected. "
               "It is not a measure of biological importance.")

    st.subheader("How this app works")
    a, b = st.columns(2)
    with a:
        st.markdown(
            '<div class="card"><div class="small">Layer 1</div><h3>Evidence Search</h3>'
            "Look up records that already exist in the dataset using any one field "
            "or a combination of Disease, Gene, Variant and Drug.</div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="card"><div class="small">Layer 2</div><h3>ML Prediction and Ranking</h3>'
            "A trained model scores how similar a new combination is to the "
            "dataset's positive examples. Predictions are never added to the "
            "evidence records.</div>",
            unsafe_allow_html=True,
        )
    st.write("")
    disclaimer()


def page_search():
    hero("Evidence Search", "Retrieve existing dataset records. Fill in only the fields you know.")

    mode = st.radio("Search mode", ["Pick from lists", "Type to search (partial match)"],
                    horizontal=True)
    scope = data
    if mode == "Pick from lists":
        cols = st.columns(4)
        for i, col in enumerate(INPUT_FEATURES):
            with cols[i]:
                choice = st.selectbox(col, ["Any"] + sorted(scope[col].unique()), key=f"s_{col}")
            if choice != "Any":
                scope = scope[scope[col] == choice]
        st.caption("Each list only shows values that still exist after your earlier choices.")
    else:
        cols = st.columns(4)
        typed = [cols[i].text_input(col, key=f"t_{col}") for i, col in enumerate(INPUT_FEATURES)]
        exact = st.checkbox("Require exact match")
        scope = search_biological_records(data, *typed, exact_match=exact)

    label_filter = st.radio("Association label", ["All", "Positive", "Negative"], horizontal=True)
    if label_filter != "All":
        scope = scope[scope["Association_Label"] == (1 if label_filter == "Positive" else 0)]

    results = scope[DISPLAY_COLUMNS]
    st.divider()

    if results.empty:
        st.warning("No matching record found in the current dataset. This does not mean "
                   "the relationship does not exist biologically.")
        return

    m = st.columns(5)
    m[0].metric("Matching records", f"{len(results):,}")
    m[1].metric("Genes", results["Gene"].nunique())
    m[2].metric("Drugs", results["Drug"].nunique())
    m[3].metric("Diseases", results["Disease"].nunique())
    m[4].metric("Positive share", f"{results['Association_Label'].mean():.0%}")

    tab_table, tab_charts = st.tabs(["Records", "Charts"])
    with tab_table:
        show = results.copy()
        show["Association_Label"] = show["Association_Label"].map(label_name)
        show["Drug_Is_Combination"] = show["Drug_Is_Combination"].astype(bool)
        st.dataframe(show, hide_index=True, width="stretch", height=420)
        st.download_button("Download results as CSV",
                           results.to_csv(index=False).encode("utf-8"),
                           "evidence_search_results.csv", "text/csv")
    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            counts = results["Association_Label"].map(label_name).value_counts()
            fig = px.pie(names=counts.index, values=counts.values, hole=0.5,
                         color=counts.index, color_discrete_map=LABEL_COLORS,
                         title="Labels in these results")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch", key="sr_pie")
        with c2:
            top = results["Drug"].value_counts().head(10).sort_values()
            fig = px.bar(x=top.values, y=top.index, orientation="h",
                         title="Drugs in these results", labels={"x": "Records", "y": ""})
            fig.update_traces(marker_color=ACCENT)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch", key="sr_bar")
    st.caption("These are records present in the dataset, not independently verified "
               "clinical evidence or approval status.")


def page_prediction():
    hero("ML Prediction", "Score one Disease-Gene-Variant-Drug combination. All four fields are required.")
    st.button("Load a random example from the dataset", on_click=example_picker,
              args=("pred",))
    with st.form("prediction_form"):
        disease, gene, variant, drug = four_inputs("pred")
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        show_prediction(predict_association(disease, gene, variant, drug), "pred")
    disclaimer()


def page_ranking():
    hero("Candidate Ranking", "Edit the table, add rows with the + at the bottom, then rank by predicted score.")
    starter = (data[data["Split"] == "test"].sample(6, random_state=3)[INPUT_FEATURES]
               .reset_index(drop=True))
    edited = st.data_editor(starter, num_rows="dynamic", width="stretch", key="rank_editor")

    if st.button("Rank candidates", type="primary"):
        rows, skipped = [], 0
        for _, r in edited.iterrows():
            if any(pd.isna(r[c]) or not str(r[c]).strip() for c in INPUT_FEATURES):
                skipped += 1
                continue
            rows.append(predict_association(r["Disease"], r["Gene"], r["Variant"], r["Drug"]))
        if skipped:
            st.info(f"{skipped} incomplete row(s) were skipped.")
        if not rows:
            st.error("Add at least one complete candidate.")
            return

        ranked = pd.DataFrame(rows).sort_values("Predicted_Association_Score", ascending=False)
        ranked = ranked.reset_index(drop=True)
        ranked["Ranking"] = ranked.index + 1
        ranked["Predicted_Label"] = ranked["Predicted_Label"].map(label_name)
        ranked["Warnings"] = ranked["Warnings"].apply(lambda w: "; ".join(w))

        ranked["Candidate"] = ranked["Drug"] + " | " + ranked["Gene"] + " | " + ranked["Disease"]
        fig = px.bar(ranked.sort_values("Predicted_Association_Score"),
                     x="Predicted_Association_Score", y="Candidate", orientation="h",
                     color="Predicted_Label", color_discrete_map=LABEL_COLORS,
                     range_x=[0, 1], title="Predicted association score")
        fig.update_layout(height=120 + 45 * len(ranked), margin=dict(l=10, r=10, t=50, b=10),
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch", key="rank_chart")

        table = ranked.drop(columns="Candidate")[
            ["Ranking", "Disease", "Gene", "Variant", "Drug", "Predicted_Label",
             "Predicted_Association_Score", "Warnings"]]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download ranked results as CSV",
                           table.to_csv(index=False).encode("utf-8"),
                           "candidate_ranking_results.csv", "text/csv")
    st.caption("Ranking shows which combinations look most similar to positive training "
               "examples. It does not identify the clinically most effective drug.")


def page_comparison():
    hero("Evidence Comparison", "See the model's prediction next to any matching records in the dataset.")
    st.button("Load a random example from the dataset", on_click=example_picker,
              args=("cmp",))
    with st.form("comparison_form"):
        disease, gene, variant, drug = four_inputs("cmp")
        submitted = st.form_submit_button("Compare", type="primary")
    if submitted:
        left, right = st.columns(2)
        with left:
            st.subheader("ML prediction")
            show_prediction(predict_association(disease, gene, variant, drug), "cmp")
        with right:
            st.subheader("Existing dataset records")
            if not all([disease, gene, variant, drug]):
                st.error("Please provide all four fields.")
            else:
                found = search_biological_records(data, disease, gene, variant, drug,
                                                  exact_match=True)
                if found.empty:
                    st.info("No matching record found in the current dataset. This does "
                            "not mean the relationship does not exist biologically.")
                else:
                    show = found[DISPLAY_COLUMNS].copy()
                    show["Association_Label"] = show["Association_Label"].map(label_name)
                    st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Predictions and dataset records are kept separate. A prediction is "
                   "never treated as an established fact.")
    disclaimer()


def page_model_info():
    hero("Model Information", "How the model was evaluated, what it relies on, and what it cannot tell you.")
    metadata = load_json_if_exists(METADATA_PATH)
    report = load_csv_if_exists(EVALUATION_REPORT_PATH)
    importance = load_csv_if_exists(IMPORTANCE_PATH)

    if metadata:
        m = st.columns(3)
        m[0].metric("Selected model", str(metadata.get("model_name", "n/a")).replace("_", " "))
        metrics = metadata.get("selected_model_metrics", {})
        if "roc_auc" in metrics:
            m[1].metric("ROC-AUC (test set)", f"{float(metrics['roc_auc']):.3f}")
        if "f1_score" in metrics:
            m[2].metric("F1-score (test set)", f"{float(metrics['f1_score']):.3f}")

    tab_perf, tab_expl, tab_meta, tab_lim = st.tabs(
        ["Performance", "Explainability", "Metadata", "Limitations"])

    with tab_perf:
        if report is None:
            st.warning("model_evaluation_report.csv was not found.")
        else:
            wanted = [c for c in ["accuracy", "precision", "recall", "f1_score",
                                  "specificity", "roc_auc", "pr_auc",
                                  "pr_auc_average_precision"] if c in report.columns]
            long = report.melt(id_vars="model_name", value_vars=wanted,
                               var_name="metric", value_name="value")
            fig = px.bar(long, x="metric", y="value", color="model_name", barmode="group",
                         title="Test-set metrics by model", range_y=[0, 1])
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch", key="mi_perf")
            st.dataframe(report[["model_name"] + wanted], hide_index=True, width="stretch")
            st.caption("The baseline always predicts the most common label, so its accuracy "
                       "(about 75%) is the number the real models need to beat.")

    with tab_expl:
        if importance is None:
            st.warning("explainability_permutation_importance.csv was not found.")
        else:
            imp = importance.sort_values("mean_importance_roc_auc_drop")
            fig = px.bar(imp, x="mean_importance_roc_auc_drop", y="feature", orientation="h",
                         error_x="std_importance", title="Permutation feature importance",
                         labels={"mean_importance_roc_auc_drop": "Drop in ROC-AUC when shuffled",
                                 "feature": ""})
            fig.update_traces(marker_color=ACCENT)
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch", key="mi_imp")
            st.write("A longer bar means the model leans on that field more. Variant "
                     "contributes almost nothing, so the model is not learning "
                     "variant-specific patterns. This describes a statistical pattern in "
                     "this dataset, not a biological cause.")

    with tab_meta:
        if metadata:
            st.json(metadata)
        else:
            st.warning("model_metadata.json was not found.")

    with tab_lim:
        st.markdown(
            "- The model was trained on one dataset and evaluated on one train/test split.\n"
            "- A high score means a combination resembles the positive training examples. "
            "It is not proof of clinical effectiveness, safety, or approval.\n"
            "- Label 0 does not necessarily mean a drug is ineffective.\n"
            "- Values never seen in training carry no learned information; the app warns "
            "when you enter one.\n"
            "- Retrieved dataset records are not independently verified evidence."
        )


# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Evidence Search": page_search,
    "ML Prediction": page_prediction,
    "Candidate Ranking": page_ranking,
    "Evidence Comparison": page_comparison,
    "Model Information": page_model_info,
}

with st.sidebar:
    st.markdown("### BIO AI")
    st.caption("Disease-Gene-Variant-Drug evidence and prediction")
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(f"Dataset: {len(data):,} records")

PAGES[choice]()
