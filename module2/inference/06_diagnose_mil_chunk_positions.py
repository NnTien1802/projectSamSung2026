from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from module2.inference.headline_integrity_pipeline import (
    HeadlineIntegrityPipeline,
)


REGRESSION_PATH = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "regression_v1"
    / "consistency_regression_samples.csv"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "mil_chunk_position_diagnostic_v1"
)

DETAIL_PATH = (
    OUTPUT_DIR
    / "chunk_position_diagnostic.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "chunk_position_diagnostic.json"
)


# ============================================================
# CONFIG
# ============================================================

CHUNK_TOKENS = 120
CHUNK_OVERLAP = 24

CHUNK_STEP = (
    CHUNK_TOKENS
    -
    CHUNK_OVERLAP
)

MAX_CHUNKS = 6

TOLERANCE = 0.005


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


def error_stats(
    saved,
    predicted,
):

    saved = np.asarray(
        saved,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    errors = np.abs(
        saved
        -
        predicted
    )

    return {
        "mean_abs_error":
            float(
                errors.mean()
            ),

        "median_abs_error":
            float(
                np.median(
                    errors
                )
            ),

        "p95_abs_error":
            float(
                np.percentile(
                    errors,
                    95,
                )
            ),

        "max_abs_error":
            float(
                errors.max()
            ),

        "within_tolerance":
            int(
                (
                    errors
                    <=
                    TOLERANCE
                ).sum()
            ),

        "total":
            int(
                len(
                    errors
                )
            ),

        "pass":
            bool(
                errors.max()
                <=
                TOLERANCE
            ),
    }


def unique_preserve_order(
    values,
):

    seen = set()
    result = []

    for value in values:

        value = int(
            value
        )

        if value not in seen:

            result.append(
                value
            )

            seen.add(
                value
            )

    return result


# ============================================================
# TOKENIZE ARTICLE
# ============================================================

def tokenize_article(
    pipeline,
    content,
):

    # Avoid the cosmetic HuggingFace >512 warning.
    old_max_length = (
        pipeline.tokenizer.model_max_length
    )

    try:

        pipeline.tokenizer.model_max_length = (
            10**9
        )

        article_ids = (
            pipeline.tokenizer(
                content,
                add_special_tokens=False,
                truncation=False,
            )[
                "input_ids"
            ]
        )

    finally:

        pipeline.tokenizer.model_max_length = (
            old_max_length
        )

    return article_ids


# ============================================================
# RAW WINDOW STARTS
# ============================================================

def build_raw_starts(
    token_count,
):

    if token_count <= 0:

        return []

    return list(
        range(
            0,
            token_count,
            CHUNK_STEP,
        )
    )


# ============================================================
# POSITION POLICIES
# ============================================================

def current_floor_indices(
    token_count,
    target_chunk_count,
):

    raw = build_raw_starts(
        token_count
    )

    if len(raw) <= target_chunk_count:

        return raw

    indices = np.linspace(
        0,
        len(raw) - 1,
        num=target_chunk_count,
        dtype=int,
    )

    return [
        raw[
            int(index)
        ]
        for index in indices
    ]


def rounded_indices(
    token_count,
    target_chunk_count,
):

    raw = build_raw_starts(
        token_count
    )

    if len(raw) <= target_chunk_count:

        return raw

    positions = np.linspace(
        0,
        len(raw) - 1,
        num=target_chunk_count,
    )

    indices = np.rint(
        positions
    ).astype(
        int
    )

    indices = np.clip(
        indices,
        0,
        len(raw) - 1,
    )

    return [
        raw[
            int(index)
        ]
        for index in indices
    ]


def ceil_indices(
    token_count,
    target_chunk_count,
):

    raw = build_raw_starts(
        token_count
    )

    if len(raw) <= target_chunk_count:

        return raw

    positions = np.linspace(
        0,
        len(raw) - 1,
        num=target_chunk_count,
    )

    indices = np.ceil(
        positions
    ).astype(
        int
    )

    indices = np.clip(
        indices,
        0,
        len(raw) - 1,
    )

    return [
        raw[
            int(index)
        ]
        for index in indices
    ]


def direct_uniform_floor(
    token_count,
    target_chunk_count,
):

    if target_chunk_count <= 1:

        return [
            0
        ]

    max_start = max(
        0,
        token_count
        -
        CHUNK_TOKENS,
    )

    starts = np.linspace(
        0,
        max_start,
        num=target_chunk_count,
        dtype=int,
    )

    return unique_preserve_order(
        starts.tolist()
    )


def direct_uniform_round(
    token_count,
    target_chunk_count,
):

    if target_chunk_count <= 1:

        return [
            0
        ]

    max_start = max(
        0,
        token_count
        -
        CHUNK_TOKENS,
    )

    starts = np.rint(
        np.linspace(
            0,
            max_start,
            num=target_chunk_count,
        )
    ).astype(
        int
    )

    return unique_preserve_order(
        starts.tolist()
    )


def direct_uniform_full_range_floor(
    token_count,
    target_chunk_count,
):

    if target_chunk_count <= 1:

        return [
            0
        ]

    max_start = max(
        0,
        token_count
        -
        1,
    )

    starts = np.linspace(
        0,
        max_start,
        num=target_chunk_count,
        dtype=int,
    )

    return unique_preserve_order(
        starts.tolist()
    )


def step120_floor_indices(
    token_count,
    target_chunk_count,
):

    raw = list(
        range(
            0,
            token_count,
            CHUNK_TOKENS,
        )
    )

    if len(raw) <= target_chunk_count:

        return raw

    indices = np.linspace(
        0,
        len(raw) - 1,
        num=target_chunk_count,
        dtype=int,
    )

    return [
        raw[
            int(index)
        ]
        for index in indices
    ]


def step120_round_indices(
    token_count,
    target_chunk_count,
):

    raw = list(
        range(
            0,
            token_count,
            CHUNK_TOKENS,
        )
    )

    if len(raw) <= target_chunk_count:

        return raw

    indices = np.rint(
        np.linspace(
            0,
            len(raw) - 1,
            num=target_chunk_count,
        )
    ).astype(
        int
    )

    indices = np.clip(
        indices,
        0,
        len(raw) - 1,
    )

    return [
        raw[
            int(index)
        ]
        for index in indices
    ]


POLICIES = {
    "current_step96_floor":
        current_floor_indices,

    "step96_round":
        rounded_indices,

    "step96_ceil":
        ceil_indices,

    "direct_uniform_maxstart_floor":
        direct_uniform_floor,

    "direct_uniform_maxstart_round":
        direct_uniform_round,

    "direct_uniform_fullrange_floor":
        direct_uniform_full_range_floor,

    "step120_floor":
        step120_floor_indices,

    "step120_round":
        step120_round_indices,
}


# ============================================================
# PREDICT WITH EXPLICIT STARTS
# ============================================================

@torch.inference_mode()
def predict_with_starts(
    pipeline,
    title,
    article_ids,
    starts,
):

    headline_ids = (
        pipeline.tokenizer(
            title,
            add_special_tokens=False,
            truncation=False,
        )[
            "input_ids"
        ]
    )

    batch_input_ids = []
    batch_attention = []
    batch_token_types = []

    valid_starts = []

    for start in starts:

        start = int(
            start
        )

        chunk_ids = article_ids[
            start:
            start
            +
            CHUNK_TOKENS
        ]

        if not chunk_ids:

            continue

        (
            input_ids,
            attention_mask,
            token_type_ids,
        ) = pipeline._build_mil_pair(
            headline_ids=headline_ids,
            chunk_ids=chunk_ids,
        )

        batch_input_ids.append(
            input_ids
        )

        batch_attention.append(
            attention_mask
        )

        batch_token_types.append(
            token_type_ids
        )

        valid_starts.append(
            start
        )

    if not batch_input_ids:

        raise RuntimeError(
            "No valid chunks generated."
        )

    input_ids_tensor = torch.tensor(
        batch_input_ids,
        dtype=torch.long,
        device=pipeline.device,
    )

    attention_tensor = torch.tensor(
        batch_attention,
        dtype=torch.long,
        device=pipeline.device,
    )

    token_type_tensor = torch.tensor(
        batch_token_types,
        dtype=torch.long,
        device=pipeline.device,
    )

    with torch.amp.autocast(
        device_type=pipeline.device.type,
        enabled=pipeline.use_amp,
    ):

        outputs = pipeline.mil_model(
            input_ids=input_ids_tensor,
            attention_mask=attention_tensor,
            token_type_ids=token_type_tensor,
        )

    logits = (
        outputs.logits
        .float()
    )

    # This aggregation was confirmed by Step 04
    # to be the closest to the original artifact.
    chunk_scores = (
        logits[
            :,
            1
        ]
        -
        logits[
            :,
            0
        ]
    )

    bag_score = torch.max(
        chunk_scores
    )

    probability = float(
        torch.sigmoid(
            bag_score
        )
        .cpu()
        .item()
    )

    max_chunk_index = int(
        torch.argmax(
            chunk_scores
        )
        .cpu()
        .item()
    )

    return {
        "probability":
            probability,

        "actual_chunk_count":
            int(
                len(
                    valid_starts
                )
            ),

        "starts":
            valid_starts,

        "max_chunk_index":
            max_chunk_index,

        "max_chunk_start":
            int(
                valid_starts[
                    max_chunk_index
                ]
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
        "INTEGRATION STEP 05"
    )
    print(
        "DIAGNOSE MAX-MIL CHUNK POSITIONS"
    )
    print("=" * 80)

    print()
    print(
        "VALIDATION REGRESSION SAMPLES ONLY."
    )

    print(
        "NO TRAINING."
    )

    print(
        "NO TEST."
    )

    print(
        "NO MODEL OR THRESHOLD CHANGES."
    )

    print()
    print(
        f"Chunk tokens : "
        f"{CHUNK_TOKENS}"
    )

    print(
        f"Overlap      : "
        f"{CHUNK_OVERLAP}"
    )

    print(
        f"Raw step     : "
        f"{CHUNK_STEP}"
    )

    print(
        f"Tolerance    : "
        f"{TOLERANCE:.6f}"
    )

    if not REGRESSION_PATH.exists():

        raise FileNotFoundError(
            REGRESSION_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD SAMPLE
    # ========================================================

    frame = pd.read_csv(
        REGRESSION_PATH
    )

    required = {
        "pair_id",
        "label",
        "headline_title",
        "anchor_content",
        "prob_mil",
        "original_article_token_count",
        "original_chunk_count",
    }

    missing = (
        required
        -
        set(
            frame.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Regression CSV missing columns: "
            f"{sorted(missing)}"
        )

    print()
    print(
        f"Samples: "
        f"{len(frame):,}"
    )

    # ========================================================
    # LOAD PIPELINE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD UNIFIED PIPELINE"
    )
    print("=" * 80)

    pipeline = (
        HeadlineIntegrityPipeline()
    )

    # ========================================================
    # RUN
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RUN POSITION POLICIES"
    )
    print("=" * 80)

    records = []

    for index, row in enumerate(
        frame.itertuples(
            index=False
        ),
        start=1,
    ):

        article_ids = tokenize_article(
            pipeline,
            row.anchor_content,
        )

        token_count = len(
            article_ids
        )

        original_token_count = int(
            row.original_article_token_count
        )

        original_chunk_count = int(
            row.original_chunk_count
        )

        if (
            token_count
            !=
            original_token_count
        ):

            raise RuntimeError(
                f"{row.pair_id}: "
                f"token count changed: "
                f"{token_count} vs "
                f"{original_token_count}"
            )

        raw_starts = build_raw_starts(
            token_count
        )

        record = {
            "pair_id":
                row.pair_id,

            "label":
                int(
                    row.label
                ),

            "saved_prob_mil":
                float(
                    row.prob_mil
                ),

            "token_count":
                token_count,

            "original_chunk_count":
                original_chunk_count,

            "raw_step96_chunk_count":
                int(
                    len(
                        raw_starts
                    )
                ),

            "raw_step96_starts":
                json.dumps(
                    raw_starts
                ),
        }

        print()
        print(
            f"[{index:02d}/{len(frame):02d}] "
            f"{row.pair_id}"
        )

        print(
            f"  tokens="
            f"{token_count} | "
            f"original_chunks="
            f"{original_chunk_count} | "
            f"raw_step96="
            f"{len(raw_starts)}"
        )

        for (
            policy_name,
            policy_function,
        ) in POLICIES.items():

            starts = (
                policy_function(
                    token_count,
                    original_chunk_count,
                )
            )

            result = (
                predict_with_starts(
                    pipeline=pipeline,
                    title=row.headline_title,
                    article_ids=article_ids,
                    starts=starts,
                )
            )

            probability = float(
                result[
                    "probability"
                ]
            )

            error = abs(
                probability
                -
                float(
                    row.prob_mil
                )
            )

            record[
                f"{policy_name}_prob"
            ] = probability

            record[
                f"{policy_name}_error"
            ] = error

            record[
                f"{policy_name}_starts"
            ] = json.dumps(
                result[
                    "starts"
                ]
            )

            record[
                f"{policy_name}_max_start"
            ] = (
                result[
                    "max_chunk_start"
                ]
            )

            print(
                f"  {policy_name:34s} "
                f"p={probability:.6f} "
                f"err={error:.6f} "
                f"starts="
                f"{result['starts']}"
            )

        records.append(
            record
        )

    detail = pd.DataFrame(
        records
    )

    detail.to_csv(
        DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SUMMARIZE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "POSITION POLICY RESULTS"
    )
    print("=" * 80)

    summary = {}

    for policy_name in POLICIES:

        probability_column = (
            f"{policy_name}_prob"
        )

        stats = error_stats(
            saved=(
                detail[
                    "saved_prob_mil"
                ]
            ),
            predicted=(
                detail[
                    probability_column
                ]
            ),
        )

        summary[
            policy_name
        ] = stats

        print()
        print(
            policy_name
        )

        print(
            f"  Mean abs error : "
            f"{stats['mean_abs_error']:.9f}"
        )

        print(
            f"  Median error   : "
            f"{stats['median_abs_error']:.9f}"
        )

        print(
            f"  P95 abs error  : "
            f"{stats['p95_abs_error']:.9f}"
        )

        print(
            f"  Max abs error  : "
            f"{stats['max_abs_error']:.9f}"
        )

        print(
            f"  Within tol     : "
            f"{stats['within_tolerance']}/"
            f"{stats['total']}"
        )

        print(
            f"  RESULT         : "
            f"{'PASS' if stats['pass'] else 'FAIL'}"
        )

    # ========================================================
    # RANKING
    # ========================================================

    ranking = sorted(
        summary.items(),
        key=lambda item: (
            item[
                1
            ][
                "mean_abs_error"
            ],
            item[
                1
            ][
                "max_abs_error"
            ],
        ),
    )

    print()
    print("=" * 80)
    print(
        "POSITION POLICY RANKING"
    )
    print("=" * 80)

    print()

    for rank, (
        policy_name,
        stats,
    ) in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{policy_name:34s} "
            f"MAE="
            f"{stats['mean_abs_error']:.9f} "
            f"MAX="
            f"{stats['max_abs_error']:.9f} "
            f"PASS="
            f"{stats['pass']}"
        )

    best_policy = (
        ranking[
            0
        ][
            0
        ]
    )

    best_stats = (
        ranking[
            0
        ][
            1
        ]
    )

    # ========================================================
    # WORST CASES
    # ========================================================

    best_error_column = (
        f"{best_policy}_error"
    )

    best_probability_column = (
        f"{best_policy}_prob"
    )

    best_starts_column = (
        f"{best_policy}_starts"
    )

    worst = (
        detail
        .sort_values(
            best_error_column,
            ascending=False,
        )
        .head(10)
    )

    print()
    print("=" * 80)
    print(
        "WORST CASES FOR BEST POLICY"
    )
    print("=" * 80)

    print()

    for _, row in worst.iterrows():

        print(
            f"{row['pair_id']} | "
            f"saved="
            f"{row['saved_prob_mil']:.6f} | "
            f"candidate="
            f"{row[best_probability_column]:.6f} | "
            f"error="
            f"{row[best_error_column]:.6f}"
        )

        print(
            f"  starts="
            f"{row[best_starts_column]}"
        )

        print(
            f"  raw_step96="
            f"{row['raw_step96_starts']}"
        )

    # ========================================================
    # FOCUS ON PREVIOUSLY BAD CASES
    # ========================================================

    current_error_column = (
        "current_step96_floor_error"
    )

    previously_bad = (
        detail[
            detail[
                current_error_column
            ]
            >
            TOLERANCE
        ]
        .copy()
    )

    print()
    print("=" * 80)
    print(
        "PREVIOUSLY FAILING SAMPLES"
    )
    print("=" * 80)

    print()

    if previously_bad.empty:

        print(
            "None."
        )

    else:

        for _, row in previously_bad.iterrows():

            print(
                f"{row['pair_id']}"
            )

            print(
                f"  saved="
                f"{row['saved_prob_mil']:.6f}"
            )

            for policy_name in POLICIES:

                print(
                    f"  {policy_name:34s} "
                    f"err="
                    f"{row[f'{policy_name}_error']:.6f} "
                    f"starts="
                    f"{row[f'{policy_name}_starts']}"
                )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage":
            "integration_step_05_mil_chunk_position_diagnostic",

        "validation_only":
            True,

        "training_performed":
            False,

        "test_used":
            False,

        "chunk_tokens":
            CHUNK_TOKENS,

        "chunk_overlap":
            CHUNK_OVERLAP,

        "chunk_step":
            CHUNK_STEP,

        "max_chunks":
            MAX_CHUNKS,

        "tolerance":
            TOLERANCE,

        "samples":
            int(
                len(
                    detail
                )
            ),

        "policies":
            summary,

        "ranking":
            [
                {
                    "rank":
                        int(
                            rank
                        ),

                    "policy":
                        policy_name,

                    "mean_abs_error":
                        stats[
                            "mean_abs_error"
                        ],

                    "max_abs_error":
                        stats[
                            "max_abs_error"
                        ],

                    "pass":
                        stats[
                            "pass"
                        ],
                }

                for rank, (
                    policy_name,
                    stats,
                )
                in enumerate(
                    ranking,
                    start=1,
                )
            ],

        "best_policy":
            best_policy,

        "best_policy_pass":
            bool(
                best_stats[
                    "pass"
                ]
            ),
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "DIAGNOSIS"
    )
    print("=" * 80)

    print()
    print(
        f"Best policy: "
        f"{best_policy}"
    )

    print(
        f"Mean abs error: "
        f"{best_stats['mean_abs_error']:.9f}"
    )

    print(
        f"Max abs error : "
        f"{best_stats['max_abs_error']:.9f}"
    )

    if best_stats[
        "pass"
    ]:

        print()
        print(
            "CHUNK POSITION POLICY IDENTIFIED."
        )

        print(
            "The best policy reproduces all sampled "
            "saved MIL probabilities within tolerance."
        )

        print()
        print(
            "Do not edit the pipeline yet."
        )

        print(
            "Use this policy in the next step, then "
            "rerun the complete regression check."
        )

    else:

        print()
        print(
            "NO POSITION POLICY PASSED."
        )

        print(
            "Next step must inspect the exact Step 17 "
            "cache-generation source code / cached chunk "
            "token IDs rather than guessing further."
        )

    print()
    print(
        f"Detail:\n"
        f"  {DETAIL_PATH}"
    )

    print(
        f"Report:\n"
        f"  {REPORT_PATH}"
    )

    print()
    print(
        "INTEGRATION STEP 05 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()