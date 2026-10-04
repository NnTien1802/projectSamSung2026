from pathlib import Path
import csv
import gc
import json
import time

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_v1"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.jsonl"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.jsonl"
)


# ------------------------------------------------------------
# Chưa sử dụng test ở bước tuning baseline.
# ------------------------------------------------------------

TEST_PATH = (
    DATA_DIR
    / "test.jsonl"
)


# ============================================================
# OUTPUT
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_baselines"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_baselines"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT FIELD
# ============================================================

# Dùng natural trước.
#
# Sau này có thể đổi thành:
# article_segmented
#
# để làm ablation study.
TEXT_FIELD = "article_natural"


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 20260918


# ============================================================
# BASELINE 1 CONFIG
# WORD TF-IDF + LOGISTIC REGRESSION
# ============================================================

WORD_MAX_FEATURES = 120_000

WORD_MIN_DF = 3

WORD_NGRAM_RANGE = (1, 2)


# ============================================================
# BASELINE 2 CONFIG
# CHAR TF-IDF + LINEAR SVM
# ============================================================

# char_wb chỉ tạo character n-gram bên trong
# word boundary.
#
# Nó phù hợp hơn char thuần cho spelling error
# và giảm bớt số lượng feature không cần thiết.

CHAR_MAX_FEATURES = 120_000

CHAR_MIN_DF = 5

CHAR_NGRAM_RANGE = (3, 5)


# ============================================================
# LOAD JSONL
# ============================================================

def load_dataset(
    path,
    text_field,
    keep_metadata=False
):

    texts = []

    labels = []

    metadata = []

    print(
        f"\nLoading:\n{path}"
    )

    start_time = time.time()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            text = row.get(
                text_field,
                ""
            )

            if text is None:
                text = ""

            texts.append(
                str(text)
            )

            labels.append(
                int(
                    row["label"]
                )
            )

            if keep_metadata:

                metadata.append({
                    "source_id":
                        row.get(
                            "source_id",
                            ""
                        ),

                    "variant_id":
                        row.get(
                            "variant_id",
                            ""
                        ),

                    "label":
                        int(
                            row[
                                "label"
                            ]
                        ),

                    "severity":
                        row.get(
                            "severity",
                            "unknown"
                        ),

                    "error_count":
                        int(
                            row.get(
                                "error_count",
                                0
                            )
                        ),
                })

    labels = np.asarray(
        labels,
        dtype=np.int8
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Records: "
        f"{len(texts):,}"
    )

    print(
        f"Clean: "
        f"{np.sum(labels == 0):,}"
    )

    print(
        f"Noisy: "
        f"{np.sum(labels == 1):,}"
    )

    print(
        f"Load time: "
        f"{elapsed:.2f}s"
    )

    if keep_metadata:

        return (
            texts,
            labels,
            metadata
        )

    return (
        texts,
        labels
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    scores,
    metadata
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "NO_ERROR",
            "HAS_ERROR",
        ],
        output_dict=True,
        zero_division=0
    )

    # ========================================================
    # Ranking metrics
    # ========================================================

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    average_precision = (
        average_precision_score(
            y_true,
            scores
        )
    )

    # ========================================================
    # Recall by severity
    # ========================================================
    #
    # Severity chỉ tồn tại có ý nghĩa cho positive/noisy.
    #
    # Vì vậy metric phù hợp ở đây là:
    #
    # Recall(severity)
    #
    # chứ không phải F1 riêng của severity.
    # ========================================================

    severity_stats = {}

    for severity in [
        "minor",
        "moderate",
        "severe",
    ]:

        indices = [
            i
            for i, item in enumerate(
                metadata
            )
            if (
                item[
                    "label"
                ]
                == 1
                and item[
                    "severity"
                ]
                == severity
            )
        ]

        if not indices:

            severity_stats[
                severity
            ] = {
                "count": 0,
                "recall": None,
            }

            continue

        severity_true = y_true[
            indices
        ]

        severity_pred = y_pred[
            indices
        ]

        severity_recall = (
            recall_score(
                severity_true,
                severity_pred,
                pos_label=1,
                zero_division=0
            )
        )

        severity_stats[
            severity
        ] = {
            "count":
                len(indices),

            "recall":
                float(
                    severity_recall
                ),
        }

    results = {
        "model":
            model_name,

        "accuracy":
            float(
                accuracy
            ),

        "error_precision":
            float(
                precision
            ),

        "error_recall":
            float(
                recall
            ),

        "error_f1":
            float(
                f1
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "roc_auc":
            float(
                roc_auc
            ),

        "average_precision":
            float(
                average_precision
            ),

        "confusion_matrix": {
            "tn":
                int(
                    cm[0, 0]
                ),

            "fp":
                int(
                    cm[0, 1]
                ),

            "fn":
                int(
                    cm[1, 0]
                ),

            "tp":
                int(
                    cm[1, 1]
                ),
        },

        "severity_recall":
            severity_stats,

        "classification_report":
            report,
    }

    return results


# ============================================================
# PRINT RESULT
# ============================================================

def print_results(
    results
):

    print("\n" + "=" * 80)

    print(
        results[
            "model"
        ]
    )

    print("=" * 80)

    print(
        f"Accuracy        : "
        f"{results['accuracy']:.6f}"
    )

    print(
        f"ERROR Precision : "
        f"{results['error_precision']:.6f}"
    )

    print(
        f"ERROR Recall    : "
        f"{results['error_recall']:.6f}"
    )

    print(
        f"ERROR F1        : "
        f"{results['error_f1']:.6f}"
    )

    print(
        f"Macro F1        : "
        f"{results['macro_f1']:.6f}"
    )

    print(
        f"ROC-AUC         : "
        f"{results['roc_auc']:.6f}"
    )

    print(
        f"AveragePrecision: "
        f"{results['average_precision']:.6f}"
    )

    cm = results[
        "confusion_matrix"
    ]

    print("\nConfusion Matrix")

    print(
        f"TN: {cm['tn']:,}"
    )

    print(
        f"FP: {cm['fp']:,}"
    )

    print(
        f"FN: {cm['fn']:,}"
    )

    print(
        f"TP: {cm['tp']:,}"
    )

    print(
        "\nRecall by severity"
    )

    for severity, values in (
        results[
            "severity_recall"
        ].items()
    ):

        recall = values[
            "recall"
        ]

        count = values[
            "count"
        ]

        if recall is None:

            print(
                f"  {severity:10s}: "
                f"N/A"
            )

        else:

            print(
                f"  {severity:10s}: "
                f"{recall:.6f} "
                f"(n={count:,})"
            )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    path,
    metadata,
    predictions,
    scores
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "source_id",
            "variant_id",
            "label",
            "prediction",
            "score",
            "severity",
            "error_count",
        ])

        for (
            item,
            prediction,
            score
        ) in zip(
            metadata,
            predictions,
            scores
        ):

            writer.writerow([
                item[
                    "source_id"
                ],

                item[
                    "variant_id"
                ],

                item[
                    "label"
                ],

                int(
                    prediction
                ),

                float(
                    score
                ),

                item[
                    "severity"
                ],

                item[
                    "error_count"
                ],
            ])


# ============================================================
# BASELINE 1
# ============================================================

def train_word_logistic_regression(
    train_texts,
    train_labels,
    validation_texts,
    validation_labels,
    validation_metadata
):

    print("\n" + "#" * 80)
    print("BASELINE 1")
    print("WORD TF-IDF + LOGISTIC REGRESSION")
    print("#" * 80)

    # ========================================================
    # VECTORIZER
    # ========================================================

    vectorizer = TfidfVectorizer(
        analyzer="word",

        ngram_range=
            WORD_NGRAM_RANGE,

        min_df=
            WORD_MIN_DF,

        max_df=
            0.995,

        max_features=
            WORD_MAX_FEATURES,

        lowercase=True,

        # Quan trọng với tiếng Việt:
        # giữ token 1 ký tự như:
        # ở, a...
        token_pattern=
            r"(?u)\b\w+\b",

        sublinear_tf=True,

        norm="l2",

        dtype=np.float32,
    )

    print(
        "\nFitting word TF-IDF..."
    )

    start = time.time()

    X_train = (
        vectorizer.fit_transform(
            train_texts
        )
    )

    X_validation = (
        vectorizer.transform(
            validation_texts
        )
    )

    print(
        f"TF-IDF time: "
        f"{time.time() - start:.2f}s"
    )

    print(
        f"Vocabulary: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"Train matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Validation matrix: "
        f"{X_validation.shape}"
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    model = LogisticRegression(
        C=1.0,

        solver="saga",

        max_iter=500,

        tol=1e-3,

        random_state=
            RANDOM_SEED,

        verbose=1,
    )

    print(
        "\nTraining Logistic Regression..."
    )

    start = time.time()

    model.fit(
        X_train,
        train_labels
    )

    train_seconds = (
        time.time()
        - start
    )

    print(
        f"\nTraining time: "
        f"{train_seconds:.2f}s"
    )

    # ========================================================
    # PREDICT
    # ========================================================

    predictions = model.predict(
        X_validation
    )

    scores = model.predict_proba(
        X_validation
    )[:, 1]

    # ========================================================
    # EVALUATE
    # ========================================================

    results = evaluate_model(
        model_name=
            "word_tfidf_logistic_regression",

        y_true=
            validation_labels,

        y_pred=
            predictions,

        scores=
            scores,

        metadata=
            validation_metadata,
    )

    results[
        "training_seconds"
    ] = float(
        train_seconds
    )

    results[
        "text_field"
    ] = TEXT_FIELD

    results[
        "tfidf_config"
    ] = {
        "analyzer":
            "word",

        "ngram_range":
            list(
                WORD_NGRAM_RANGE
            ),

        "min_df":
            WORD_MIN_DF,

        "max_features":
            WORD_MAX_FEATURES,
    }

    print_results(
        results
    )

    # ========================================================
    # SAVE
    # ========================================================

    vectorizer_path = (
        MODEL_DIR
        / "baseline1_word_tfidf.joblib"
    )

    model_path = (
        MODEL_DIR
        / "baseline1_logistic_regression.joblib"
    )

    result_path = (
        RESULT_DIR
        / "baseline1_validation_metrics.json"
    )

    prediction_path = (
        RESULT_DIR
        / "baseline1_validation_predictions.csv"
    )

    print(
        "\nSaving Baseline 1..."
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
        compress=3
    )

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    save_predictions(
        prediction_path,
        validation_metadata,
        predictions,
        scores
    )

    print(
        f"Vectorizer : "
        f"{vectorizer_path}"
    )

    print(
        f"Model      : "
        f"{model_path}"
    )

    print(
        f"Metrics    : "
        f"{result_path}"
    )

    print(
        f"Predictions: "
        f"{prediction_path}"
    )

    # ========================================================
    # FREE MEMORY
    # ========================================================

    del X_train
    del X_validation
    del vectorizer
    del model
    del predictions
    del scores

    gc.collect()

    return results


# ============================================================
# BASELINE 2
# ============================================================

def train_char_linear_svm(
    train_texts,
    train_labels,
    validation_texts,
    validation_labels,
    validation_metadata
):

    print("\n" + "#" * 80)
    print("BASELINE 2")
    print("CHAR TF-IDF + LINEAR SVM")
    print("#" * 80)

    # ========================================================
    # CHARACTER TF-IDF
    # ========================================================

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",

        ngram_range=
            CHAR_NGRAM_RANGE,

        min_df=
            CHAR_MIN_DF,

        max_features=
            CHAR_MAX_FEATURES,

        lowercase=True,

        sublinear_tf=True,

        norm="l2",

        dtype=np.float32,
    )

    print(
        "\nFitting character TF-IDF..."
    )

    start = time.time()

    X_train = (
        vectorizer.fit_transform(
            train_texts
        )
    )

    X_validation = (
        vectorizer.transform(
            validation_texts
        )
    )

    print(
        f"TF-IDF time: "
        f"{time.time() - start:.2f}s"
    )

    print(
        f"Vocabulary: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"Train matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Validation matrix: "
        f"{X_validation.shape}"
    )

    # ========================================================
    # LINEAR SVM
    # ========================================================

    model = LinearSVC(
        C=1.0,

        dual="auto",

        max_iter=5000,

        tol=1e-4,

        random_state=
            RANDOM_SEED,
    )

    print(
        "\nTraining LinearSVC..."
    )

    start = time.time()

    model.fit(
        X_train,
        train_labels
    )

    train_seconds = (
        time.time()
        - start
    )

    print(
        f"Training time: "
        f"{train_seconds:.2f}s"
    )

    # ========================================================
    # PREDICT
    # ========================================================

    predictions = model.predict(
        X_validation
    )

    scores = model.decision_function(
        X_validation
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    results = evaluate_model(
        model_name=
            "char_tfidf_linear_svm",

        y_true=
            validation_labels,

        y_pred=
            predictions,

        scores=
            scores,

        metadata=
            validation_metadata,
    )

    results[
        "training_seconds"
    ] = float(
        train_seconds
    )

    results[
        "text_field"
    ] = TEXT_FIELD

    results[
        "tfidf_config"
    ] = {
        "analyzer":
            "char_wb",

        "ngram_range":
            list(
                CHAR_NGRAM_RANGE
            ),

        "min_df":
            CHAR_MIN_DF,

        "max_features":
            CHAR_MAX_FEATURES,
    }

    print_results(
        results
    )

    # ========================================================
    # SAVE
    # ========================================================

    vectorizer_path = (
        MODEL_DIR
        / "baseline2_char_tfidf.joblib"
    )

    model_path = (
        MODEL_DIR
        / "baseline2_linear_svm.joblib"
    )

    result_path = (
        RESULT_DIR
        / "baseline2_validation_metrics.json"
    )

    prediction_path = (
        RESULT_DIR
        / "baseline2_validation_predictions.csv"
    )

    print(
        "\nSaving Baseline 2..."
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
        compress=3
    )

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    save_predictions(
        prediction_path,
        validation_metadata,
        predictions,
        scores
    )

    print(
        f"Vectorizer : "
        f"{vectorizer_path}"
    )

    print(
        f"Model      : "
        f"{model_path}"
    )

    print(
        f"Metrics    : "
        f"{result_path}"
    )

    print(
        f"Predictions: "
        f"{prediction_path}"
    )

    # ========================================================
    # FREE MEMORY
    # ========================================================

    del X_train
    del X_validation
    del vectorizer
    del model
    del predictions
    del scores

    gc.collect()

    return results


# ============================================================
# COMPARISON
# ============================================================

def save_comparison(
    baseline1,
    baseline2
):

    comparison = {
        "text_field":
            TEXT_FIELD,

        "evaluation_split":
            "validation",

        "baseline1":
            baseline1,

        "baseline2":
            baseline2,
    }

    path = (
        RESULT_DIR
        / "baseline_comparison.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            comparison,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Model':35s}"
        f"{'Accuracy':>12s}"
        f"{'Error F1':>12s}"
        f"{'Macro F1':>12s}"
    )

    print("-" * 71)

    print(
        f"{'Word TF-IDF + Logistic Regression':35s}"
        f"{baseline1['accuracy']:12.4f}"
        f"{baseline1['error_f1']:12.4f}"
        f"{baseline1['macro_f1']:12.4f}"
    )

    print(
        f"{'Char TF-IDF + LinearSVC':35s}"
        f"{baseline2['accuracy']:12.4f}"
        f"{baseline2['error_f1']:12.4f}"
        f"{baseline2['macro_f1']:12.4f}"
    )

    print(
        f"\nComparison saved:\n{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VN-NEWSSPELL - CLASSICAL BASELINES")
    print("=" * 80)

    print(
        f"\nText field: "
        f"{TEXT_FIELD}"
    )

    print(
        "\nEvaluation policy:"
        "\n  train      -> model training"
        "\n  validation -> evaluation/tuning"
        "\n  test       -> NOT USED YET"
    )

    # ========================================================
    # LOAD TRAIN
    # ========================================================

    (
        train_texts,
        train_labels
    ) = load_dataset(
        TRAIN_PATH,
        TEXT_FIELD,
        keep_metadata=False
    )

    # ========================================================
    # LOAD VALIDATION
    # ========================================================

    (
        validation_texts,
        validation_labels,
        validation_metadata
    ) = load_dataset(
        VALIDATION_PATH,
        TEXT_FIELD,
        keep_metadata=True
    )

    # ========================================================
    # BASELINE 1
    # ========================================================

    baseline1 = (
        train_word_logistic_regression(
            train_texts,
            train_labels,
            validation_texts,
            validation_labels,
            validation_metadata
        )
    )

    # ========================================================
    # BASELINE 2
    # ========================================================

    baseline2 = (
        train_char_linear_svm(
            train_texts,
            train_labels,
            validation_texts,
            validation_labels,
            validation_metadata
        )
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    save_comparison(
        baseline1,
        baseline2
    )

    print("\n" + "=" * 80)
    print("BASELINE TRAINING FINISHED")
    print("=" * 80)

    print(
        "\nTest set was NOT used."
    )


if __name__ == "__main__":
    main()