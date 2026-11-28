import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SpamGuard | Intelligent Message Analysis",
    page_icon="SG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
GRAPH_DIR = ROOT / "graphs"
RESULT_DIR = ROOT / "results"

FINAL_METRIC_LABELS = (
    ("Test accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1_score"),
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc"),
)

HISTORICAL_BASELINE_FILES = {"model_metrics.csv", "predictions.csv"}

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "naive_bayes_model.pkl"
    vectorizer_path = MODEL_DIR / "count_vectorizer.pkl"
    config_path = MODEL_DIR / "model_config.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration not found: {config_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    config = joblib.load(config_path)

    if not isinstance(model, MultinomialNB):
        raise ValueError("Saved classifier is not a Multinomial Naive Bayes model.")
    if not isinstance(config, dict):
        raise ValueError("Saved model configuration must be a dictionary.")

    try:
        threshold = float(config["decision_threshold"])
        configured_alpha = float(config["alpha"])
        configured_vocabulary_size = int(config["vocabulary_size"])
        configured_algorithm = config["algorithm"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Saved model configuration is incomplete or invalid.") from exc

    if configured_algorithm != "Multinomial Naive Bayes":
        raise ValueError("Saved model configuration names an unexpected algorithm.")
    if not 0 <= threshold <= 1:
        raise ValueError("Saved decision_threshold must be between 0 and 1.")
    if not np.isclose(float(model.alpha), configured_alpha):
        raise ValueError("Saved model alpha does not match model_config.pkl.")
    if (
        len(vectorizer.vocabulary_) != configured_vocabulary_size
        or model.n_features_in_ != configured_vocabulary_size
    ):
        raise ValueError(
            "Saved model, vectorizer and model configuration have different vocabularies."
        )
    if set(model.classes_) != {"ham", "spam"}:
        raise ValueError("Saved model must contain the ham and spam classes.")

    return model, vectorizer, config


def load_final_evaluation(model, vectorizer, decision_threshold: float):
    metrics_path = RESULT_DIR / "final_test_metrics.csv"
    predictions_path = RESULT_DIR / "final_test_predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Canonical test metrics not found: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Canonical test predictions not found: {predictions_path}"
        )

    metrics_df = pd.read_csv(metrics_path)
    predictions_df = pd.read_csv(predictions_path)
    required_prediction_columns = {
        "message",
        "actual_label",
        "predicted_label",
        "spam_probability",
    }
    if not required_prediction_columns.issubset(predictions_df.columns):
        raise ValueError("Canonical test predictions have an unexpected format.")
    if not {"metric", "value"}.issubset(metrics_df.columns):
        raise ValueError("Canonical test metrics have an unexpected format.")
    if predictions_df.empty:
        raise ValueError("Canonical test predictions are empty.")

    actual = predictions_df["actual_label"].astype(str).to_numpy()
    stored_predictions = predictions_df["predicted_label"].astype(str).to_numpy()
    spam_probabilities = pd.to_numeric(
        predictions_df["spam_probability"], errors="raise"
    ).to_numpy(dtype=float)

    if not set(actual).issubset({"ham", "spam"}):
        raise ValueError("Canonical test predictions contain unexpected labels.")
    if not np.isfinite(spam_probabilities).all() or (
        (spam_probabilities < 0) | (spam_probabilities > 1)
    ).any():
        raise ValueError("Canonical test predictions contain invalid probabilities.")

    model_spam_index = list(model.classes_).index("spam")
    model_probabilities = model.predict_proba(
        vectorizer.transform(predictions_df["message"].fillna(""))
    )[:, model_spam_index]
    if not np.allclose(
        model_probabilities, spam_probabilities, rtol=0, atol=1e-12
    ):
        raise ValueError(
            "Canonical test probabilities do not match the saved model and vectorizer."
        )

    predicted = np.where(
        spam_probabilities >= decision_threshold, "spam", "ham"
    )
    if not np.array_equal(predicted, stored_predictions):
        raise ValueError(
            "Canonical predictions do not match the saved decision threshold."
        )

    y_true = (actual == "spam").astype(int)
    y_pred = (predicted == "spam").astype(int)
    calculated_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, spam_probabilities),
        "pr_auc": average_precision_score(y_true, spam_probabilities),
    }

    stored_metrics = metrics_df.set_index("metric")["value"].to_dict()
    for metric, value in calculated_metrics.items():
        if metric not in stored_metrics or not np.isclose(
            float(stored_metrics[metric]), value, rtol=0, atol=1e-12
        ):
            raise ValueError(
                f"Canonical metric {metric!r} does not match the saved predictions."
            )

    matrix = confusion_matrix(actual, predicted, labels=["ham", "spam"])
    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual ham", "Actual spam"],
        columns=["Predicted ham", "Predicted spam"],
    )
    return calculated_metrics, confusion_df, len(predictions_df)


try:
    model, vectorizer, model_config = load_model()
    decision_threshold = float(model_config["decision_threshold"])
    MODEL_READY = True
    MODEL_ERROR = None
except Exception as exc:
    model, vectorizer, model_config = None, None, None
    decision_threshold = None
    MODEL_READY = False
    MODEL_ERROR = str(exc)

decision_threshold_label = (
    f"{decision_threshold:.2f}"
    if decision_threshold is not None
    else "unavailable"
)

try:
    if not MODEL_READY:
        raise RuntimeError("The saved model configuration is unavailable.")
    FINAL_METRICS, FINAL_CONFUSION_MATRIX, FINAL_TEST_COUNT = (
        load_final_evaluation(model, vectorizer, decision_threshold)
    )
    EVALUATION_READY = True
    EVALUATION_ERROR = None
except Exception as exc:
    FINAL_METRICS, FINAL_CONFUSION_MATRIX, FINAL_TEST_COUNT = {}, None, 0
    EVALUATION_READY = False
    EVALUATION_ERROR = str(exc)


# ============================================================
# CUSTOM DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #090E1C;
        --panel: #111A2E;
        --panel-2: #151F36;
        --border: #263653;
        --purple: #A78BFA;
        --purple-bright: #8B5CF6;
        --text: #F3F6FC;
        --muted: #9BAAC2;
        --green: #34D399;
        --red: #FB7185;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(91, 63, 171, 0.12), transparent 27%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }

    .block-container {
        max-width: 1220px;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.45px;
    }

    p, label, span, div {
        font-family: 'DM Sans', sans-serif;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 7px 0 24px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 25px;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 25px;
        font-weight: 700;
        letter-spacing: -1.1px;
        color: #FFFFFF;
    }

    .brand-accent { color: var(--purple); }

    .top-status {
        color: #B9A7FF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
    }

    .hero {
        padding: 48px 48px 45px 48px;
        border: 1px solid #34436B;
        border-radius: 23px;
        background:
            radial-gradient(circle at 88% 12%, rgba(167,139,250,0.24), transparent 32%),
            linear-gradient(125deg, #192B55 0%, #17233E 55%, #111A2E 100%);
        margin: 18px 0 34px 0;
    }

    .eyebrow {
        color: #C4B5FD;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 17px;
    }

    .hero-title {
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(34px, 5vw, 53px);
        font-weight: 700;
        letter-spacing: -2px;
        line-height: 1.12;
        margin: 0 0 17px 0;
    }

    .hero-copy {
        color: #C5D0E4;
        font-size: 15px;
        line-height: 1.85;
        max-width: 760px;
        margin: 0;
    }

    .section-kicker {
        color: var(--purple);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 8px 0 8px 0;
    }

    .section-title {
        color: #F8FAFC;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 600;
        letter-spacing: -0.8px;
        margin: 0 0 7px 0;
    }

    .section-copy {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.75;
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(145deg, #17243D, #111A2E);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px 21px;
        min-height: 126px;
    }

    .metric-label {
        color: #A9B8D0;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 29px;
        font-weight: 700;
        letter-spacing: -0.8px;
    }

    .metric-note {
        color: #B7A4FF;
        font-size: 11px;
        margin-top: 5px;
    }

    .surface {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 23px;
    }

    .result-card {
        border-radius: 16px;
        padding: 23px 25px;
        margin: 18px 0;
        border: 1px solid;
    }

    .result-spam {
        background: rgba(244, 63, 94, 0.08);
        border-color: rgba(251, 113, 133, 0.42);
    }

    .result-ham {
        background: rgba(16, 185, 129, 0.07);
        border-color: rgba(52, 211, 153, 0.42);
    }

    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .result-spam .result-title { color: var(--red); }
    .result-ham .result-title { color: var(--green); }

    .result-copy {
        color: #B8C5D8;
        font-size: 13px;
        line-height: 1.7;
    }

    .stTextArea textarea {
        background: #0D1629 !important;
        color: #F8FAFC !important;
        border: 1px solid #344563 !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        padding: 15px !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--purple) !important;
        box-shadow: 0 0 0 1px var(--purple) !important;
    }

    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(100deg, #7438E8, #9667F6);
        color: #FFFFFF;
        border: 0;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: linear-gradient(100deg, #6630D1, #8050E8);
        color: #FFFFFF;
        border: 0;
    }

    div[data-testid="stTabs"] button {
        color: #AAB8CE;
        font-weight: 600;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #C4B5FD;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    .small-note {
        color: #8292AC;
        font-size: 11px;
        line-height: 1.7;
    }

    .footer {
        text-align: center;
        color: #71819C;
        font-size: 11px;
        line-height: 1.9;
        padding: 25px 0 5px 0;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.18) !important;
        margin: 30px 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS FOR RESEARCH FILES
# ============================================================

GRAPH_TITLES = {
    "alpha_comparison": "Laplace smoothing / alpha comparison",
    "smoothing_visualization": "Effect of Laplace smoothing on word probabilities",
    "prior_comparison": "Class prior probability comparison",
    "prior_probability_comparison": "Class prior probability comparison",
    "threshold_analysis": "Decision threshold analysis",
    "calibration_curve": "Probability calibration",
    "bootstrap_f1_distribution": "Validation message-group bootstrap F1 distribution",
    "final_test_confusion_matrix": "Final independent-test confusion matrix",
    "refined_error_terms": "Terms associated with classification errors",
    "feature_analysis": "Words most strongly associated with each class",
    "conditional_independence": "Conditional independence analysis",
    "independence_analysis": "Conditional independence analysis",
    "error_distribution": "Distribution of classification errors",
}

GRAPH_DESCRIPTIONS = {
    "alpha_comparison": (
        "Compares validation performance across smoothing values; the independent "
        "test set is not used. "
        "The experiment selected alpha = 0.1 based on the reported validation PR-AUC."
    ),
    "threshold_analysis": (
        "Validation-only analysis of how the decision threshold affects precision, "
        "recall and F1-score; the independent test set is not used."
    ),
    "calibration_curve": (
        "Validation-set predicted spam probabilities versus observed spam frequencies."
    ),
    "bootstrap_f1_distribution": (
        "Shows validation F1 uncertainty from resampling unique message-text groups; "
        "duplicate rows are kept together."
    ),
    "final_test_confusion_matrix": (
        "Summarises correct and incorrect predictions on the independent test set."
    ),
    "refined_error_terms": (
        "Highlights terms and phrases found in false-positive and false-negative examples."
    ),
    "feature_analysis": (
        "Ranks words by their association with the spam and legitimate-message classes."
    ),
    "prior_comparison": (
        "Compares the dataset's observed class priors with an equal-prior alternative."
    ),
    "prior_probability_comparison": (
        "Full-dataset descriptive prior comparison; not held-out performance evidence."
    ),
}


def readable_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem in GRAPH_TITLES:
        return GRAPH_TITLES[stem]
    return re.sub(r"[_\-]+", " ", path.stem).strip().title()


def graph_description(path: Path) -> str:
    return GRAPH_DESCRIPTIONS.get(
        path.stem.lower(),
        "Experimental visualisation generated during the B4 research workflow.",
    )


@st.cache_data
def list_graph_files():
    if not GRAPH_DIR.exists():
        return []
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        [p for p in GRAPH_DIR.iterdir() if p.is_file() and p.suffix.lower() in allowed],
        key=lambda p: (p.stem.lower() not in GRAPH_TITLES, p.name.lower()),
    )


@st.cache_data
def list_result_files():
    if not RESULT_DIR.exists():
        return []
    return sorted(
        [p for p in RESULT_DIR.glob("*.csv") if p.is_file()],
        key=lambda p: p.name.lower(),
    )


@st.cache_data
def read_csv_file(path_string: str):
    path = Path(path_string)
    return pd.read_csv(path)


def metric_card(label: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP NAVIGATION AND HERO
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">Spam<span class="brand-accent">Guard</span>.</div>
        <div class="top-status">PROBABILISTIC TEXT CLASSIFICATION</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">B4 · Research implementation</div>
        <div class="hero-title">Understand the message.<br>Measure the risk.</div>
        <p class="hero-copy">
            SpamGuard applies Multinomial Naive Bayes to SMS text classification.
            Explore the working detector, inspect the model's independent-test
            performance, and review the probability experiments behind the system.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_READY:
    st.error(
        "The saved model could not be loaded. Check that "
        "`models/naive_bayes_model.pkl`, `models/count_vectorizer.pkl` and "
        "`models/model_config.pkl` "
        "exist in the project folder."
    )
    st.code(MODEL_ERROR or "Unknown model-loading error")
if not EVALUATION_READY:
    st.error(
        "Canonical independent-test results are unavailable or inconsistent; "
        "cached or hardcoded evaluation values will not be displayed."
    )
    st.code(EVALUATION_ERROR or "Unknown evaluation-data error")


# ============================================================
# MAIN NAVIGATION
# ============================================================

page = st.radio(
    "Main navigation",
    [
        "Overview",
        "Message Detector",
        "Research Experiments",
        "Model Evaluation",
        "Methodology",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown('<div class="section-kicker">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Performance on unseen messages.</div>', unsafe_allow_html=True)
    if EVALUATION_READY:
        st.markdown(
            f'<div class="section-copy">Canonical final metrics are derived from and '
            f'checked against the saved independent-test predictions '
            f'({FINAL_TEST_COUNT} messages).</div>',
            unsafe_allow_html=True,
        )
        metric_columns = st.columns(4)
        for column, (label, key) in zip(metric_columns, FINAL_METRIC_LABELS[:4]):
            with column:
                metric_card(
                    label.upper(),
                    f"{FINAL_METRICS[key] * 100:.2f}%",
                    "Independent test evaluation",
                )
    else:
        st.warning("Performance cards are hidden until canonical test files are valid.")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="section-kicker">THE SYSTEM</div>', unsafe_allow_html=True)
        st.markdown("### From message to prediction")
        st.markdown(
            """
            <div class="surface">
                <b>01 · Text representation</b>
                <p class="small-note">CountVectorizer converts message text into word-count features.</p>
                <hr>
                <b>02 · Probability estimation</b>
                <p class="small-note">Multinomial Naive Bayes estimates class probabilities using learned word frequencies and class priors.</p>
                <hr>
                <b>03 · Classification</b>
                <p class="small-note">Predict spam when its estimated probability is at least the configured threshold; otherwise predict ham.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Current decision rule: spam if P(spam) ≥ {decision_threshold_label}; "
            "ham otherwise."
        )

    with right:
        st.markdown('<div class="section-kicker">EXPERIMENT SNAPSHOT</div>', unsafe_allow_html=True)
        st.markdown("### Dataset and model")
        st.markdown(
            """
            <div class="surface">
                <p><span class="small-note">TOTAL DATASET RECORDS</span><br><b>5,574</b></p>
                <p><span class="small-note">FINAL TRAINING RECORDS</span><br><b>4,459</b></p>
                <p><span class="small-note">INDEPENDENT TEST RECORDS</span><br><b>1,115</b></p>
                <p><span class="small-note">VOCABULARY SIZE</span><br><b>7,673</b></p>
                <p><span class="small-note">SELECTED SMOOTHING VALUE</span><br><b>α = 0.1</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "The alpha value was selected during validation. Final performance figures "
        "are from the separate independent test evaluation. The deployed decision "
        f"threshold is {decision_threshold_label}."
    )


# ============================================================
# MESSAGE DETECTOR
# ============================================================

elif page == "Message Detector":
    st.markdown('<div class="section-kicker">LIVE CLASSIFICATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analyse an SMS message.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Enter a message to obtain a prediction from the saved '
        'Multinomial Naive Bayes model.</div>',
        unsafe_allow_html=True,
    )

    with st.form("message_analysis", clear_on_submit=False):
        message = st.text_area(
            "MESSAGE INPUT",
            placeholder="Paste the SMS message you want to analyse...",
            height=165,
            max_chars=10000,
        )
        submitted = st.form_submit_button("Analyse message", use_container_width=True)

    if submitted:
        if not message.strip():
            st.warning("Enter a message before starting the analysis.")
        elif not MODEL_READY:
            st.error("The classifier is unavailable because the saved model did not load.")
        else:
            with st.spinner("Analysing message patterns..."):
                message_vector = vectorizer.transform([message])
                probabilities = model.predict_proba(message_vector)[0]
                classes = list(model.classes_)

                if "spam" not in classes or "ham" not in classes:
                    st.error("The saved model does not contain the expected ham and spam classes.")
                    st.stop()

                spam_probability = float(probabilities[classes.index("spam")])
                ham_probability = float(probabilities[classes.index("ham")])
                prediction = (
                    "spam"
                    if spam_probability >= decision_threshold
                    else "ham"
                )

            st.caption(
                f"Decision rule: classify as spam when P(spam) ≥ "
                f"{decision_threshold:.2f}; otherwise classify as ham."
            )

            if prediction == "spam":
                st.markdown(
                    """
                    <div class="result-card result-spam">
                        <div class="result-title">Spam message detected</div>
                        <div class="result-copy">The model classified this message as likely unwanted communication.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result-card result-ham">
                        <div class="result-title">Legitimate message</div>
                        <div class="result-copy">The model classified this message as likely normal communication.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            probability_columns = st.columns(2)
            with probability_columns[0]:
                st.markdown("**Spam probability**")
                st.progress(spam_probability)
                st.caption(f"{spam_probability * 100:.2f}%")
            with probability_columns[1]:
                st.markdown("**Legitimate probability**")
                st.progress(ham_probability)
                st.caption(f"{ham_probability * 100:.2f}%")

            st.caption(
                "These are model-estimated probabilities, not guarantees that a message "
                "is safe, fraudulent or authentic."
            )
            with st.expander("View analysed message"):
                st.write(message)


# ============================================================
# RESEARCH EXPERIMENTS
# ============================================================

elif page == "Research Experiments":
    st.markdown('<div class="section-kicker">EXPERIMENTAL RESULTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The experiments behind the model.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">These visualisations are loaded directly from the project '
        '`graphs` folder. Add new experiment plots to that folder and they will appear here.</div>',
        unsafe_allow_html=True,
    )

    graph_files = list_graph_files()

    if not graph_files:
        st.warning(
            "No graph images were found. Check that your experiment plots are saved "
            "inside the project's `graphs` folder."
        )
    else:
        graph_names = [readable_name(path) for path in graph_files]
        selected_graph_name = st.selectbox("Select an experiment", graph_names)
        selected_path = graph_files[graph_names.index(selected_graph_name)]

        st.markdown(
            f'<div class="surface"><b>{html.escape(selected_graph_name)}</b>'
            f'<p class="small-note">{html.escape(graph_description(selected_path))}</p></div>',
            unsafe_allow_html=True,
        )
        st.image(str(selected_path), use_container_width=True)

        st.markdown("### Browse all experiment figures")
        for start in range(0, len(graph_files), 2):
            cols = st.columns(2)
            for offset, column in enumerate(cols):
                index = start + offset
                if index >= len(graph_files):
                    break
                path = graph_files[index]
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{readable_name(path)}**")
                        st.image(str(path), use_container_width=True)
                        st.caption(graph_description(path))

    st.divider()
    st.markdown("### Experimental result files")
    st.caption(
        "Final-test files are explicitly named final_test_*. Other experiment files "
        "are validation or training-only outputs as documented in the report; "
        "historical alpha-1 baseline files are identified below."
    )
    result_files = list_result_files()

    if result_files:
        result_names = [p.name for p in result_files]
        selected_result = st.selectbox("Choose a results table", result_names)
        selected_result_path = RESULT_DIR / selected_result
        if selected_result in HISTORICAL_BASELINE_FILES:
            st.warning(
                "Historical baseline only: generated with alpha=1.0 and the "
                "default 0.5 decision threshold. This is not the canonical final "
                "independent-test result; see final_test_metrics.csv and "
                "final_test_predictions.csv."
            )
        elif selected_result.startswith("final_test_"):
            st.info("Canonical independent-test evaluation output.")
        else:
            st.info(
                "Experiment output, not a final independent-test result. "
                "See the report for the data partition and settings."
            )
        try:
            result_df = read_csv_file(str(selected_result_path))
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download this results table",
                data=selected_result_path.read_bytes(),
                file_name=selected_result_path.name,
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not read {selected_result}: {exc}")
    else:
        st.info("No CSV result files were found in the project's `results` folder.")


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":
    st.markdown('<div class="section-kicker">INDEPENDENT TEST EVALUATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How well does the classifier perform?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The figures below are from the final independent test '
        f'evaluation containing {FINAL_TEST_COUNT} messages.</div>',
        unsafe_allow_html=True,
    )

    if EVALUATION_READY:
        metric_columns = st.columns(3)
        for index, (name, key) in enumerate(FINAL_METRIC_LABELS):
            with metric_columns[index % len(metric_columns)]:
                metric_card(
                    name.upper(),
                    f"{FINAL_METRICS[key] * 100:.2f}%",
                    "Independent test evaluation",
                )

        st.write("")
        st.markdown("### Confusion matrix")
        st.dataframe(FINAL_CONFUSION_MATRIX, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Correct ham",
            f'{FINAL_CONFUSION_MATRIX.loc["Actual ham", "Predicted ham"]:,}',
        )
        c2.metric(
            "Ham flagged as spam",
            f'{FINAL_CONFUSION_MATRIX.loc["Actual ham", "Predicted spam"]:,}',
        )
        c3.metric(
            "Spam missed",
            f'{FINAL_CONFUSION_MATRIX.loc["Actual spam", "Predicted ham"]:,}',
        )
        c4.metric(
            "Correct spam",
            f'{FINAL_CONFUSION_MATRIX.loc["Actual spam", "Predicted spam"]:,}',
        )

        st.caption(
            "Rows represent actual labels and columns represent predicted labels. "
            "A false negative is a spam message predicted as ham; a false positive is "
            "a legitimate message predicted as spam."
        )
    else:
        st.warning(
            "Evaluation metrics and confusion matrix are hidden because the "
            "canonical outputs failed validation."
        )

    if EVALUATION_READY and GRAPH_DIR.exists() and (GRAPH_DIR / "final_test_confusion_matrix.png").exists():
        st.markdown("### Confusion matrix visualisation")
        st.image(
            str(GRAPH_DIR / "final_test_confusion_matrix.png"),
            use_container_width=True,
        )

    st.info(
        "The validation experiments were used for model selection. The independent "
        "test set was kept separate for the final performance evaluation. "
        f"The final spam threshold is {decision_threshold_label}."
    )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":
    st.markdown('<div class="section-kicker">RESEARCH METHODOLOGY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">A probabilistic approach to SMS classification.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The project studies how word frequencies and class '
        'probabilities can be used to distinguish spam from legitimate SMS messages.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Dataset preparation")
    st.write(
        "The dataset contains 5,574 labelled SMS messages. The completed workflow "
        "uses 4,459 records for final model training and 1,115 records for independent "
        "testing. A separate validation split was used during model selection."
    )

    st.markdown("### 2. Text representation")
    st.write(
        "CountVectorizer converts each message into a numerical vector of token counts. "
        "The final vectorizer is fitted on the 4,459 training-plus-validation messages "
        "and reused for test and new input. The implementation uses CountVectorizer "
        "defaults: lowercase=True, token_pattern='(?u)\\b\\w\\w+\\b', "
        "strip_accents=None, and no stop-word filtering. One-character tokens are "
        "omitted; no explicit Unicode normalization or accent stripping is configured."
    )

    st.markdown("### 3. Multinomial Naive Bayes")
    st.write(
        "The classifier estimates the probability of a class given the observed word "
        "counts. It uses class prior probabilities and word likelihoods learned from "
        "the labelled training data."
    )
    st.latex(
        r"P(c\mid d) \propto P(c)\prod_{i=1}^{n}P(w_i\mid c)^{x_i}"
    )
    st.write(
        "Here, c is the message class, d is the message representation, "
        "P(c) is the class prior, P(wᵢ | c) is the likelihood of a word given "
        "the class, n is the number of vocabulary terms, and xᵢ is the count "
        "of that word in the message."
    )

    st.markdown("### 4. Laplace smoothing")
    st.write(
        "Laplace smoothing prevents a word that was not observed in a class during "
        "training from receiving a zero likelihood."
    )
    st.latex(
        r"P(w_i\mid c)=\frac{N_{ic}+\alpha}{N_c+\alpha V}"
    )
    st.write(
        "Nᵢc is the count of word i in class c, N꜀ is the total token count for "
        "class c, V is the vocabulary size, and α is the smoothing parameter. "
        "The validation alpha comparison selected α = 0.1. The final spam decision "
        f"threshold is {decision_threshold_label}. This is a precision/recall trade-off: "
        "validation recall at 0.3 is higher than at the higher-F1 thresholds around "
        "0.6–0.8; 0.3 is not claimed to maximize F1."
    )

    st.markdown("### 5. Evaluation and statistical analysis")
    st.write(
        "The project evaluates accuracy, precision, recall, F1-score, ROC-AUC and "
        "PR-AUC. Additional experiments examine class priors, decision thresholds, "
        "probability calibration, bootstrap confidence intervals and classification errors."
    )

    st.markdown("### 6. Technology stack")
    tech_columns = st.columns(3)
    with tech_columns[0]:
        st.markdown("**Machine learning**")
        st.write("Python\n\nScikit-learn\n\nMultinomial Naive Bayes")
    with tech_columns[1]:
        st.markdown("**Data and analysis**")
        st.write("Pandas\n\nNumPy\n\nMatplotlib")
    with tech_columns[2]:
        st.markdown("**Application**")
        st.write("Streamlit\n\nJoblib\n\nCountVectorizer")

    st.markdown("### Limitations")
    st.write(
        "The classifier is trained on a labelled SMS dataset. Its performance may "
        "change on messages from different sources, languages or time periods. "
        "New spam wording, unseen vocabulary, short messages and ambiguous wording "
        "can affect predictions. Model probabilities should be interpreted as estimates, "
        "not proof that a message is safe or fraudulent."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div class="footer">
        <b>SPAMGUARD</b> · INTELLIGENT MESSAGE CLASSIFICATION<br>
        B4 Research Project · Multinomial Naive Bayes<br>
        Built with Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
