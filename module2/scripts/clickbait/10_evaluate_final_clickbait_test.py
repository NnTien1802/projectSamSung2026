from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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

TEST_PATH = (
    DATA_DIR
    / "test.parquet"
)

FINAL_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "final_title_only_v1"
)

LOCK_PATH = (
    FINAL_DIR
    / "final_model_lock.json"
)

PREDICTION_PATH = (
    FINAL_DIR
    / "official_test_predictions.parquet"
)

METRICS_PATH = (
    FINAL_DIR
    / "official_test_metrics.json"
)

SOURCE_PATH = (
    FINAL_DIR
    / "official_test_by_source.csv"
)

CATEGORY_PATH = (
    FINAL_DIR
    / "official_test_by_category.csv"
)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_MODEL = (
    "bert-base-multilingual-cased"
)

EXPECTED_INPUT = (
    "title_only"
)

EXPECTED_MAX_LENGTH = 160

EXPECTED_THRESHOLD = 0.61

INFERENCE_BATCH_SIZE = 32

MIN_GROUP_SIZE = 10


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def human_time(seconds):

    seconds = int(seconds)

    hours = (
        seconds
        //
        3600
    )

    minutes = (
        (
            seconds
            %
            3600
        )
        //
        60
    )

    seconds = (
        seconds
        %
        60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# DATASET
# ============================================================

class ClickbaitTestDataset(
    Dataset
):

    def __init__(
        self,
        frame,
    ):

        self.sample_ids = (
            frame[
                "sample_id"
            ]
            .astype(str)
            .tolist()
        )

        self.titles = (
            frame[
                "title"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.labels = (
            frame[
                "label"
            ]
            .astype(int)
            .tolist()
        )

    def __len__(
        self,
    ):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index,
    ):

        return {
            "sample_id":
                self.sample_ids[
                    index
                ],

            "title":
                self.titles[
                    index
                ],

            "label":
                self.labels[
                    index
                ],
        }


class TitleCollator:

    def __init__(
        self,
        tokenizer,
        max_length,
    ):

        self.tokenizer = tokenizer

        self.max_length = int(
            max_length
        )

    def __call__(
        self,
        batch,
    ):

        sample_ids = [
            item[
                "sample_id"
            ]
            for item in batch
        ]

        titles = [
            item[
                "title"
            ]
            for item in batch
        ]

        labels = torch.tensor(
            [
                item[
                    "label"
                ]
                for item in batch
            ],
            dtype=torch.long,
        )

        encoded = self.tokenizer(
            titles,
            padding=True,
            truncation=True,
            max_length=(
                self.max_length
            ),
            return_tensors="pt",
        )

        encoded[
            "labels"
        ] = labels

        encoded[
            "sample_ids"
        ] = sample_ids

        return encoded


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

    return {
        "threshold":
            float(
                threshold
            ),

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
            float(
                roc_auc_score(
                    labels,
                    probabilities,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    labels,
                    probabilities,
                )
            ),

        "confusion_matrix": {
            "tn":
                int(
                    tn
                ),

            "fp":
                int(
                    fp
                ),

            "fn":
                int(
                    fn
                ),

            "tp":
                int(
                    tp
                ),
        },
    }


# ============================================================
# SUBGROUP METRICS
# ============================================================

def build_group_metrics(
    frame,
    group_column,
    threshold,
):

    rows = []

    for group_name, group in (
        frame.groupby(
            group_column,
            dropna=False,
        )
    ):

        if (
            len(
                group
            )
            <
            MIN_GROUP_SIZE
        ):

            continue

        labels = (
            group[
                "label"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        probabilities = (
            group[
                "prob_clickbait"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        predictions = (
            probabilities
            >=
            threshold
        ).astype(
            np.int64
        )

        normal_n = int(
            (
                labels
                ==
                0
            )
            .sum()
        )

        clickbait_n = int(
            (
                labels
                ==
                1
            )
            .sum()
        )

        macro_f1 = float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        accuracy = float(
            accuracy_score(
                labels,
                predictions,
            )
        )

        precision_clickbait = float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        )

        recall_clickbait = float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        )

        f1_clickbait = float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        )

        rows.append(
            {
                group_column:
                    str(
                        group_name
                    ),

                "n":
                    int(
                        len(
                            group
                        )
                    ),

                "normal_n":
                    normal_n,

                "clickbait_n":
                    clickbait_n,

                "clickbait_rate":
                    float(
                        labels.mean()
                    ),

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,

                "precision_clickbait":
                    precision_clickbait,

                "recall_clickbait":
                    recall_clickbait,

                "f1_clickbait":
                    f1_clickbait,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result.sort_values(
            [
                "macro_f1",
                "n",
            ],
            ascending=[
                True,
                False,
            ],
            inplace=True,
        )

    return result


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    loader,
    device,
    use_amp,
):

    model.eval()

    all_ids = []
    all_labels = []
    all_probabilities = []

    total_batches = len(
        loader
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        sample_ids = batch.pop(
            "sample_ids"
        )

        labels = (
            batch.pop(
                "labels"
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        inputs = {
            key:
                value.to(
                    device,
                    non_blocking=True,
                )

            for key, value
            in batch.items()
        }

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            outputs = model(
                **inputs
            )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )[
                :,
                1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

        all_ids.extend(
            sample_ids
        )

        all_labels.extend(
            labels_np.tolist()
        )

        all_probabilities.extend(
            probabilities.tolist()
        )

        if (
            batch_index
            %
            10
            ==
            0
            or
            batch_index
            ==
            total_batches
        ):

            print(
                f"  batch "
                f"{batch_index:,}/"
                f"{total_batches:,}"
            )

    return {
        "sample_ids":
            all_ids,

        "labels":
            np.asarray(
                all_labels,
                dtype=np.int64,
            ),

        "probabilities":
            np.asarray(
                all_probabilities,
                dtype=np.float32,
            ),
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
        "STEP 10 - OFFICIAL FINAL TEST EVALUATION"
    )
    print("=" * 80)

    print()
    print(
        "THIS IS THE OFFICIAL TEST RUN."
    )

    print(
        "NO CHECKPOINT, ARCHITECTURE OR "
        "THRESHOLD SEARCH WILL BE PERFORMED."
    )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD FINAL LOCK
    # ========================================================

    if not LOCK_PATH.exists():

        raise FileNotFoundError(
            LOCK_PATH
        )

    lock = load_json(
        LOCK_PATH
    )

    if (
        lock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_TEST"
    ):

        raise RuntimeError(
            "Final model was not properly "
            "locked before TEST."
        )

    final_model = (
        lock[
            "final_model"
        ]
    )

    architecture = (
        final_model[
            "architecture"
        ]
    )

    input_type = (
        final_model[
            "input"
        ]
    )

    max_length = int(
        final_model[
            "max_length"
        ]
    )

    threshold = float(
        final_model[
            "decision_threshold"
        ]
    )

    checkpoint_path = (
        MODULE_ROOT
        /
        final_model[
            "checkpoint"
        ]
    )

    expected_sha256 = (
        final_model[
            "checkpoint_sha256"
        ]
    )

    # ========================================================
    # ASSERT FROZEN CONFIG
    # ========================================================

    if (
        architecture
        !=
        EXPECTED_MODEL
    ):

        raise RuntimeError(
            f"Unexpected architecture: "
            f"{architecture}"
        )

    if (
        input_type
        !=
        EXPECTED_INPUT
    ):

        raise RuntimeError(
            f"Unexpected input type: "
            f"{input_type}"
        )

    if (
        max_length
        !=
        EXPECTED_MAX_LENGTH
    ):

        raise RuntimeError(
            f"Unexpected max length: "
            f"{max_length}"
        )

    if (
        abs(
            threshold
            -
            EXPECTED_THRESHOLD
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"Unexpected threshold: "
            f"{threshold}"
        )

    print()
    print(
        "FROZEN CONFIG:"
    )

    print(
        f"  Architecture : "
        f"{architecture}"
    )

    print(
        f"  Input        : "
        f"{input_type}"
    )

    print(
        f"  Max length   : "
        f"{max_length}"
    )

    print(
        f"  Threshold    : "
        f"{threshold:.2f}"
    )

    # ========================================================
    # VERIFY CHECKPOINT HASH
    # ========================================================

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    print()
    print(
        "Verifying checkpoint hash..."
    )

    actual_sha256 = (
        sha256_file(
            checkpoint_path
        )
    )

    if (
        actual_sha256
        !=
        expected_sha256
    ):

        raise RuntimeError(
            "Checkpoint SHA256 mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual  : {actual_sha256}"
        )

    print(
        "Checkpoint hash: VERIFIED"
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    if (
        device.type
        ==
        "cuda"
    ):

        print(
            "GPU   : "
            f"{torch.cuda.get_device_name(0)}"
        )

    use_amp = (
        device.type
        ==
        "cuda"
    )

    # ========================================================
    # FIRST TEST LOAD
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD OFFICIAL TEST"
    )
    print("=" * 80)

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            TEST_PATH
        )

    test = (
        pq.read_table(
            TEST_PATH,
            columns=[
                "sample_id",
                "title",
                "lead",
                "source",
                "category",
                "label",
                "label_name",
            ],
        )
        .to_pandas()
    )

    if (
        test[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "TEST contains duplicate sample_id."
        )

    test[
        "title"
    ] = (
        test[
            "title"
        ]
        .fillna("")
        .astype(str)
    )

    if (
        test[
            "title"
        ]
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "TEST contains empty title."
        )

    labels = set(
        test[
            "label"
        ].unique()
    )

    if not labels.issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"Invalid TEST labels: "
            f"{labels}"
        )

    normal_n = int(
        (
            test[
                "label"
            ]
            ==
            0
        )
        .sum()
    )

    clickbait_n = int(
        (
            test[
                "label"
            ]
            ==
            1
        )
        .sum()
    )

    print()
    print(
        f"TEST samples : "
        f"{len(test):,}"
    )

    print(
        f"NORMAL       : "
        f"{normal_n:,}"
    )

    print(
        f"CLICKBAIT    : "
        f"{clickbait_n:,}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            architecture
        )
    )

    dataset = ClickbaitTestDataset(
        test
    )

    collator = TitleCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=(
            INFERENCE_BATCH_SIZE
        ),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Loading model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            architecture,
            num_labels=2,
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if (
        "model_state_dict"
        not in
        checkpoint
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "model_state_dict."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    # ========================================================
    # OFFICIAL TEST INFERENCE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "OFFICIAL TEST INFERENCE"
    )
    print("=" * 80)

    start_time = time.time()

    inference = run_inference(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
    )

    inference_seconds = (
        time.time()
        -
        start_time
    )

    print()
    print(
        f"Inference time: "
        f"{human_time(inference_seconds)}"
    )

    # ========================================================
    # JOIN BACK TO TEST METADATA
    # ========================================================

    prediction_frame = pd.DataFrame(
        {
            "sample_id":
                inference[
                    "sample_ids"
                ],

            "inference_label":
                inference[
                    "labels"
                ],

            "prob_clickbait":
                inference[
                    "probabilities"
                ],
        }
    )

    result = (
        test
        .merge(
            prediction_frame,
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(
            result
        )
        !=
        len(
            test
        )
    ):

        raise RuntimeError(
            "TEST prediction join lost rows."
        )

    if not (
        result[
            "label"
        ]
        .to_numpy()
        ==
        result[
            "inference_label"
        ]
        .to_numpy()
    ).all():

        raise RuntimeError(
            "TEST label mismatch after inference."
        )

    result.drop(
        columns=[
            "inference_label"
        ],
        inplace=True,
    )

    # ========================================================
    # FIXED DECISION RULE
    # ========================================================

    result[
        "prediction"
    ] = (
        result[
            "prob_clickbait"
        ]
        >=
        threshold
    ).astype(
        np.int8
    )

    result[
        "prediction_name"
    ] = np.where(
        result[
            "prediction"
        ]
        ==
        1,
        "CLICKBAIT",
        "NORMAL",
    )

    result[
        "correct"
    ] = (
        result[
            "prediction"
        ]
        ==
        result[
            "label"
        ]
    )

    # ========================================================
    # OFFICIAL METRICS
    # ========================================================

    official_metrics = compute_metrics(
        labels=(
            result[
                "label"
            ]
        ),
        probabilities=(
            result[
                "prob_clickbait"
            ]
        ),
        threshold=threshold,
    )

    # ========================================================
    # SUBGROUP REPORTS
    #
    # Descriptive only.
    # They do not change the final model.
    # ========================================================

    source_df = build_group_metrics(
        frame=result,
        group_column="source",
        threshold=threshold,
    )

    category_df = build_group_metrics(
        frame=result,
        group_column="category",
        threshold=threshold,
    )

    source_df.to_csv(
        SOURCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    category_df.to_csv(
        CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    result[
        [
            "sample_id",
            "source",
            "category",
            "label",
            "label_name",
            "prob_clickbait",
            "prediction",
            "prediction_name",
            "correct",
            "title",
            "lead",
        ]
    ].to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    # ========================================================
    # SAVE OFFICIAL REPORT
    # ========================================================

    report = {
        "stage":
            "10_evaluate_final_clickbait_test",

        "task":
            "2A_clickbait_detection",

        "dataset":
            "viclickbait_v1",

        "evaluation_status":
            "OFFICIAL_TEST_COMPLETE",

        "configuration_frozen_before_test":
            True,

        "test_tuning_performed":
            False,

        "test_samples":
            int(
                len(
                    result
                )
            ),

        "test_class_distribution": {
            "normal":
                normal_n,

            "clickbait":
                clickbait_n,

            "clickbait_rate":
                float(
                    result[
                        "label"
                    ].mean()
                ),
        },

        "frozen_configuration": {
            "architecture":
                architecture,

            "input":
                input_type,

            "max_length":
                max_length,

            "decision_threshold":
                threshold,

            "checkpoint":
                str(
                    final_model[
                        "checkpoint"
                    ]
                ),

            "checkpoint_sha256":
                actual_sha256,

            "best_epoch":
                int(
                    final_model[
                        "best_epoch"
                    ]
                ),
        },

        "official_final_metrics":
            official_metrics,

        "inference": {
            "batch_size":
                INFERENCE_BATCH_SIZE,

            "device":
                str(
                    device
                ),

            "seconds":
                float(
                    inference_seconds
                ),
        },

        "methodology_note":
            (
                "TEST was evaluated using the "
                "checkpoint, architecture, input "
                "representation, max length and "
                "decision threshold frozen in Step 09. "
                "No TEST-based search or tuning was performed."
            ),
    }

    save_json(
        report,
        METRICS_PATH,
    )

    # ========================================================
    # PRINT OFFICIAL RESULT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "OFFICIAL TEST RESULT"
    )
    print("=" * 80)

    print()
    print(
        "Frozen model:"
    )

    print(
        f"  architecture : "
        f"{architecture}"
    )

    print(
        f"  input        : "
        f"{input_type}"
    )

    print(
        f"  threshold    : "
        f"{threshold:.2f}"
    )

    print()
    print(
        f"TEST samples : "
        f"{len(result):,}"
    )

    print(
        f"NORMAL       : "
        f"{normal_n:,}"
    )

    print(
        f"CLICKBAIT    : "
        f"{clickbait_n:,}"
    )

    print()
    print(
        "FINAL TITLE-ONLY MBERT:"
    )

    print(
        f"  Accuracy        : "
        f"{official_metrics['accuracy']:.6f}"
    )

    print(
        f"  Precision CLICK : "
        f"{official_metrics['precision_clickbait']:.6f}"
    )

    print(
        f"  Recall CLICK    : "
        f"{official_metrics['recall_clickbait']:.6f}"
    )

    print(
        f"  F1 CLICK        : "
        f"{official_metrics['f1_clickbait']:.6f}"
    )

    print(
        f"  Macro-F1        : "
        f"{official_metrics['macro_f1']:.6f}"
    )

    print(
        f"  ROC-AUC         : "
        f"{official_metrics['roc_auc']:.6f}"
    )

    print(
        f"  AP              : "
        f"{official_metrics['average_precision']:.6f}"
    )

    cm = (
        official_metrics[
            "confusion_matrix"
        ]
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        f"  TN={cm['tn']:,} "
        f"FP={cm['fp']:,}"
    )

    print(
        f"  FN={cm['fn']:,} "
        f"TP={cm['tp']:,}"
    )

    # ========================================================
    # LOWEST SOURCE GROUPS
    # ========================================================

    if not source_df.empty:

        print()
        print(
            "Lowest 5 sources "
            f"(n >= {MIN_GROUP_SIZE}):"
        )

        for row in (
            source_df
            .head(5)
            .itertuples(
                index=False
            )
        ):

            print(
                f"  {row.source}: "
                f"Macro-F1="
                f"{row.macro_f1:.4f} "
                f"(n={row.n})"
            )

    # ========================================================
    # LOWEST CATEGORY GROUPS
    # ========================================================

    if not category_df.empty:

        print()
        print(
            "Lowest 5 categories "
            f"(n >= {MIN_GROUP_SIZE}):"
        )

        for row in (
            category_df
            .head(5)
            .itertuples(
                index=False
            )
        ):

            print(
                f"  {row.category}: "
                f"Macro-F1="
                f"{row.macro_f1:.4f} "
                f"(n={row.n})"
            )

    print()
    print(
        "NO TEST-BASED TUNING WAS PERFORMED."
    )

    print()
    print(
        f"Predictions:\n"
        f"{PREDICTION_PATH}"
    )

    print()
    print(
        f"Metrics:\n"
        f"{METRICS_PATH}"
    )

    print()
    print(
        f"By source:\n"
        f"{SOURCE_PATH}"
    )

    print()
    print(
        f"By category:\n"
        f"{CATEGORY_PATH}"
    )

    print()
    print(
        "STEP 10 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()