from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scipy.sparse import csr_matrix, hstack

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "clickbait"
    / "viclickbait_v1"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.parquet"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.parquet"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "diagnostic_baselines_v1"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "baseline_summary.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "diagnostic_report.json"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "validation_predictions.parquet"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 20260921

THRESHOLDS = np.arange(
    0.10,
    0.901,
    0.01,
)


# ============================================================
# HELPERS
# ============================================================

def save_json(
    data,
    path,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def safe_text(
    value,
):

    if pd.isna(value):

        return ""

    return str(value)


def normalize_text(
    value,
):

    value = safe_text(
        value
    )

    value = unicodedata.normalize(
        "NFC",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def word_count(
    text,
):

    text = normalize_text(
        text
    )

    if not text:

        return 0

    return len(
        text.split()
    )


def uppercase_ratio(
    text,
):

    text = normalize_text(
        text
    )

    letters = [
        ch
        for ch in text
        if ch.isalpha()
    ]

    if not letters:

        return 0.0

    return float(
        sum(
            ch.isupper()
            for ch in letters
        )
        /
        len(letters)
    )


# ============================================================
# LOAD
# ============================================================

def load_split(
    name,
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    frame = (
        pq.read_table(
            path
        )
        .to_pandas()
    )

    required = {
        "sample_id",
        "title",
        "lead",
        "source",
        "category",
        "label",
    }

    missing = (
        required
        -
        set(frame.columns)
    )

    if missing:

        raise RuntimeError(
            f"{name}: missing columns "
            f"{sorted(missing)}"
        )

    if (
        frame[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{name}: duplicate sample_id."
        )

    frame[
        "title"
    ] = (
        frame[
            "title"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "lead"
    ] = (
        frame[
            "lead"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "source"
    ] = (
        frame[
            "source"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "category"
    ] = (
        frame[
            "category"
        ]
        .fillna("")
        .astype(str)
    )

    return frame


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    labels,
    probabilities,
    threshold,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >=
        threshold
    ).astype(
        np.int64
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    try:

        roc_auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

    except ValueError:

        roc_auc = None

    try:

        ap = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

    except ValueError:

        ap = None

    return {
        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "precision_clickbait":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall_clickbait":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1_clickbait":
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "roc_auc":
            roc_auc,

        "average_precision":
            ap,

        "confusion_matrix": {
            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),
        },
    }


def tune_threshold(
    labels,
    probabilities,
):

    best = None

    for threshold in THRESHOLDS:

        metrics = compute_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=float(
                threshold
            ),
        )

        key = (
            metrics[
                "macro_f1"
            ],

            metrics[
                "accuracy"
            ],

            -abs(
                threshold
                -
                0.5
            ),
        )

        if best is None:

            best = (
                metrics,
                key,
            )

        elif (
            key
            >
            best[1]
        ):

            best = (
                metrics,
                key,
            )

    return best[0]


# ============================================================
# SURFACE FEATURES
# ============================================================

def surface_features(
    frame,
):

    rows = []

    for row in frame.itertuples():

        title = safe_text(
            row.title
        )

        lead = safe_text(
            row.lead
        )

        rows.append(
            [
                len(title),

                word_count(
                    title
                ),

                len(lead),

                word_count(
                    lead
                ),

                int(
                    "?"
                    in
                    title
                ),

                int(
                    "!"
                    in
                    title
                ),

                int(
                    bool(
                        re.search(
                            r"\d",
                            title,
                        )
                    )
                ),

                int(
                    (
                        "..."
                        in
                        title
                    )
                    or
                    (
                        "…"
                        in
                        title
                    )
                ),

                int(
                    any(
                        quote
                        in
                        title

                        for quote in [
                            '"',
                            "'",
                            "“",
                            "”",
                            "‘",
                            "’",
                        ]
                    )
                ),

                int(
                    ":"
                    in
                    title
                ),

                int(
                    any(
                        dash
                        in
                        title

                        for dash in [
                            "-",
                            "–",
                            "—",
                        ]
                    )
                ),

                uppercase_ratio(
                    title
                ),

                int(
                    lead.strip()
                    ==
                    ""
                ),
            ]
        )

    return np.asarray(
        rows,
        dtype=np.float32,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    name,
    labels,
    probabilities,
):

    metrics_05 = (
        compute_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=0.5,
        )
    )

    tuned = tune_threshold(
        labels=labels,
        probabilities=probabilities,
    )

    print()
    print(
        f"{name}:"
    )

    print(
        f"  @0.50 Macro-F1 : "
        f"{metrics_05['macro_f1']:.6f}"
    )

    print(
        f"  @0.50 Accuracy : "
        f"{metrics_05['accuracy']:.6f}"
    )

    print(
        f"  ROC-AUC        : "
        f"{metrics_05['roc_auc']:.6f}"
    )

    print(
        f"  Tuned threshold: "
        f"{tuned['threshold']:.2f}"
    )

    print(
        f"  Tuned Macro-F1 : "
        f"{tuned['macro_f1']:.6f}"
    )

    print(
        f"  Tuned Accuracy : "
        f"{tuned['accuracy']:.6f}"
    )

    return {
        "model":
            name,

        "macro_f1_05":
            metrics_05[
                "macro_f1"
            ],

        "accuracy_05":
            metrics_05[
                "accuracy"
            ],

        "precision_clickbait_05":
            metrics_05[
                "precision_clickbait"
            ],

        "recall_clickbait_05":
            metrics_05[
                "recall_clickbait"
            ],

        "f1_clickbait_05":
            metrics_05[
                "f1_clickbait"
            ],

        "roc_auc":
            metrics_05[
                "roc_auc"
            ],

        "average_precision":
            metrics_05[
                "average_precision"
            ],

        "tuned_threshold":
            tuned[
                "threshold"
            ],

        "macro_f1_tuned":
            tuned[
                "macro_f1"
            ],

        "accuracy_tuned":
            tuned[
                "accuracy"
            ],

        "precision_clickbait_tuned":
            tuned[
                "precision_clickbait"
            ],

        "recall_clickbait_tuned":
            tuned[
                "recall_clickbait"
            ],

        "f1_clickbait_tuned":
            tuned[
                "f1_clickbait"
            ],

        "confusion_matrix_tuned":
            tuned[
                "confusion_matrix"
            ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2A - CLICKBAIT DETECTION"
    )
    print(
        "STEP 05 - DIAGNOSTIC BASELINES"
    )
    print("=" * 80)

    print()
    print(
        "TRAIN + VALIDATION ONLY."
    )

    print(
        "TEST IS NOT LOADED."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    train = load_split(
        "train",
        TRAIN_PATH,
    )

    validation = load_split(
        "validation",
        VALIDATION_PATH,
    )

    y_train = (
        train[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    y_val = (
        validation[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    print()
    print(
        f"TRAIN      : "
        f"{len(train):,}"
    )

    print(
        f"VALIDATION : "
        f"{len(validation):,}"
    )

    # ========================================================
    # RESULT CONTAINER
    # ========================================================

    rows = []

    predictions = pd.DataFrame(
        {
            "sample_id":
                validation[
                    "sample_id"
                ],

            "label":
                y_val,
        }
    )

    # ========================================================
    # 1. MAJORITY BASELINE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "1. MAJORITY BASELINE"
    )
    print("=" * 80)

    majority_probability = float(
        y_train.mean()
    )

    majority_probs = np.full(
        len(validation),
        majority_probability,
        dtype=np.float64,
    )

    result = evaluate_model(
        name="majority",
        labels=y_val,
        probabilities=(
            majority_probs
        ),
    )

    rows.append(
        result
    )

    predictions[
        "prob_majority"
    ] = majority_probs

    # ========================================================
    # 2. SOURCE-ONLY BASELINE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "2. SOURCE-ONLY BASELINE"
    )
    print("=" * 80)

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )

    source_train = encoder.fit_transform(
        train[
            [
                "source"
            ]
        ]
    )

    source_val = encoder.transform(
        validation[
            [
                "source"
            ]
        ]
    )

    source_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    source_model.fit(
        source_train,
        y_train,
    )

    source_probs = (
        source_model
        .predict_proba(
            source_val
        )[
            :,
            1
        ]
    )

    result = evaluate_model(
        name="source_only",
        labels=y_val,
        probabilities=source_probs,
    )

    rows.append(
        result
    )

    predictions[
        "prob_source_only"
    ] = source_probs

    # ========================================================
    # 3. SOURCE + CATEGORY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "3. SOURCE + CATEGORY BASELINE"
    )
    print("=" * 80)

    try:

        meta_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:

        meta_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )

    metadata_train = (
        meta_encoder
        .fit_transform(
            train[
                [
                    "source",
                    "category",
                ]
            ]
        )
    )

    metadata_val = (
        meta_encoder
        .transform(
            validation[
                [
                    "source",
                    "category",
                ]
            ]
        )
    )

    metadata_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    metadata_model.fit(
        metadata_train,
        y_train,
    )

    metadata_probs = (
        metadata_model
        .predict_proba(
            metadata_val
        )[
            :,
            1
        ]
    )

    result = evaluate_model(
        name="source_category",
        labels=y_val,
        probabilities=(
            metadata_probs
        ),
    )

    rows.append(
        result
    )

    predictions[
        "prob_source_category"
    ] = metadata_probs

    # ========================================================
    # 4. SURFACE FEATURES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "4. SURFACE FEATURE BASELINE"
    )
    print("=" * 80)

    x_surface_train = (
        surface_features(
            train
        )
    )

    x_surface_val = (
        surface_features(
            validation
        )
    )

    scaler = StandardScaler()

    x_surface_train = (
        scaler.fit_transform(
            x_surface_train
        )
    )

    x_surface_val = (
        scaler.transform(
            x_surface_val
        )
    )

    surface_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    surface_model.fit(
        x_surface_train,
        y_train,
    )

    surface_probs = (
        surface_model
        .predict_proba(
            x_surface_val
        )[
            :,
            1
        ]
    )

    result = evaluate_model(
        name="surface_features",
        labels=y_val,
        probabilities=(
            surface_probs
        ),
    )

    rows.append(
        result
    )

    predictions[
        "prob_surface"
    ] = surface_probs

    # ========================================================
    # 5. TF-IDF TITLE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "5. TF-IDF TITLE BASELINE"
    )
    print("=" * 80)

    title_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(
            1,
            2,
        ),
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents=None,
    )

    x_title_train = (
        title_vectorizer
        .fit_transform(
            train[
                "title"
            ]
        )
    )

    x_title_val = (
        title_vectorizer
        .transform(
            validation[
                "title"
            ]
        )
    )

    title_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=SEED,
        C=1.0,
    )

    title_model.fit(
        x_title_train,
        y_train,
    )

    title_probs = (
        title_model
        .predict_proba(
            x_title_val
        )[
            :,
            1
        ]
    )

    result = evaluate_model(
        name="tfidf_title",
        labels=y_val,
        probabilities=(
            title_probs
        ),
    )

    rows.append(
        result
    )

    predictions[
        "prob_tfidf_title"
    ] = title_probs

    # ========================================================
    # 6. TF-IDF TITLE + LEAD
    # ========================================================

    print()
    print("=" * 80)
    print(
        "6. TF-IDF TITLE + LEAD BASELINE"
    )
    print("=" * 80)

    train_title_lead = (
        train[
            "title"
        ]
        +
        " [SEP] "
        +
        train[
            "lead"
        ]
    )

    val_title_lead = (
        validation[
            "title"
        ]
        +
        " [SEP] "
        +
        validation[
            "lead"
        ]
    )

    pair_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(
            1,
            2,
        ),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents=None,
    )

    x_pair_train = (
        pair_vectorizer
        .fit_transform(
            train_title_lead
        )
    )

    x_pair_val = (
        pair_vectorizer
        .transform(
            val_title_lead
        )
    )

    pair_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=SEED,
        C=1.0,
    )

    pair_model.fit(
        x_pair_train,
        y_train,
    )

    pair_probs = (
        pair_model
        .predict_proba(
            x_pair_val
        )[
            :,
            1
        ]
    )

    result = evaluate_model(
        name="tfidf_title_lead",
        labels=y_val,
        probabilities=(
            pair_probs
        ),
    )

    rows.append(
        result
    )

    predictions[
        "prob_tfidf_title_lead"
    ] = pair_probs

    # ========================================================
    # SAVE
    # ========================================================

    summary = pd.DataFrame(
        rows
    )

    summary.sort_values(
        "macro_f1_tuned",
        ascending=False,
        inplace=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    # ========================================================
    # DERIVED COMPARISONS
    # ========================================================

    result_map = {
        row[
            "model"
        ]:
            row

        for row in rows
    }

    title_macro = (
        result_map[
            "tfidf_title"
        ][
            "macro_f1_tuned"
        ]
    )

    title_lead_macro = (
        result_map[
            "tfidf_title_lead"
        ][
            "macro_f1_tuned"
        ]
    )

    source_macro = (
        result_map[
            "source_only"
        ][
            "macro_f1_tuned"
        ]
    )

    surface_macro = (
        result_map[
            "surface_features"
        ][
            "macro_f1_tuned"
        ]
    )

    report = {
        "stage":
            "05_train_diagnostic_baselines",

        "task":
            "2A_clickbait_detection",

        "dataset":
            "viclickbait_v1",

        "train_rows":
            int(
                len(train)
            ),

        "validation_rows":
            int(
                len(validation)
            ),

        "test_loaded":
            False,

        "baselines":
            rows,

        "comparisons": {
            "tfidf_title_lead_minus_title_macro_f1":
                float(
                    title_lead_macro
                    -
                    title_macro
                ),

            "source_only_macro_f1":
                float(
                    source_macro
                ),

            "surface_only_macro_f1":
                float(
                    surface_macro
                ),
        },

        "interpretation_policy": {
            "source_models":
                (
                    "Diagnostic only. "
                    "Publisher identity will not "
                    "be used as a production model input."
                ),

            "surface_models":
                (
                    "Diagnostic only. Used to measure "
                    "possible shallow shortcuts."
                ),

            "test_policy":
                (
                    "TEST remains locked until "
                    "final architecture and threshold "
                    "are frozen."
                ),
        },
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 05 SUMMARY"
    )
    print("=" * 80)

    print()

    display_columns = [
        "model",
        "macro_f1_05",
        "macro_f1_tuned",
        "tuned_threshold",
        "roc_auc",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Key comparisons:"
    )

    print(
        f"  TF-IDF title Macro-F1      : "
        f"{title_macro:.6f}"
    )

    print(
        f"  TF-IDF title+lead Macro-F1 : "
        f"{title_lead_macro:.6f}"
    )

    print(
        f"  Lead delta                 : "
        f"{title_lead_macro - title_macro:+.6f}"
    )

    print(
        f"  Source-only Macro-F1       : "
        f"{source_macro:.6f}"
    )

    print(
        f"  Surface-only Macro-F1      : "
        f"{surface_macro:.6f}"
    )

    print()
    print(
        "TEST WAS NOT LOADED."
    )

    print()
    print(
        f"Summary:\n"
        f"{SUMMARY_PATH}"
    )

    print()
    print(
        f"Report:\n"
        f"{REPORT_PATH}"
    )

    print()
    print(
        f"Validation predictions:\n"
        f"{PREDICTION_PATH}"
    )

    print()
    print(
        "STEP 05 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()