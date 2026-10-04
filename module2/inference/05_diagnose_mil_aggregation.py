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
    / "mil_diagnostic_v1"
)

DETAIL_PATH = (
    OUTPUT_DIR
    / "mil_aggregation_diagnostic.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "mil_aggregation_diagnostic.json"
)


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# GET CHUNK LOGITS
# ============================================================

@torch.inference_mode()
def get_chunk_logits(
    pipeline,
    title,
    content,
):

    (
        article_ids,
        chunks,
    ) = pipeline._build_article_chunks(
        content
    )

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

    starts = []

    for chunk in chunks:

        (
            input_ids,
            attention_mask,
            token_type_ids,
        ) = pipeline._build_mil_pair(
            headline_ids=headline_ids,
            chunk_ids=(
                chunk[
                    "token_ids"
                ]
            ),
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

        starts.append(
            int(
                chunk[
                    "start_token"
                ]
            )
        )

    input_ids = torch.tensor(
        batch_input_ids,
        dtype=torch.long,
        device=pipeline.device,
    )

    attention_mask = torch.tensor(
        batch_attention,
        dtype=torch.long,
        device=pipeline.device,
    )

    token_type_ids = torch.tensor(
        batch_token_types,
        dtype=torch.long,
        device=pipeline.device,
    )

    with torch.amp.autocast(
        device_type=pipeline.device.type,
        enabled=pipeline.use_amp,
    ):

        outputs = pipeline.mil_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    return {
        "article_token_count":
            int(
                len(
                    article_ids
                )
            ),

        "chunk_count":
            int(
                len(
                    chunks
                )
            ),

        "chunk_starts":
            starts,

        "logits":
            outputs.logits.detach(),
    }


# ============================================================
# AGGREGATION VARIANTS
# ============================================================

def compute_variants(
    logits,
):

    logits_float = (
        logits
        .float()
    )

    # --------------------------------------------------------
    # Variant A
    #
    # Current unified pipeline:
    #
    # max(logit1 - logit0), then sigmoid
    # --------------------------------------------------------

    score_difference = (
        logits_float[
            :,
            1
        ]
        -
        logits_float[
            :,
            0
        ]
    )

    probability_current = float(
        torch.sigmoid(
            torch.max(
                score_difference
            )
        )
        .cpu()
        .item()
    )

    # --------------------------------------------------------
    # Variant B
    #
    # HARD MAX BAG LOGITS:
    #
    # max each class logit independently across chunks,
    # then softmax the resulting two-dimensional bag logits.
    #
    # This is the main suspected Step-18 implementation.
    # --------------------------------------------------------

    bag_logits_elementwise = (
        torch.max(
            logits_float,
            dim=0,
        )
        .values
    )

    probability_elementwise_max_logits = float(
        torch.softmax(
            bag_logits_elementwise,
            dim=-1,
        )[
            1
        ]
        .cpu()
        .item()
    )

    # --------------------------------------------------------
    # Variant C
    #
    # Convert every chunk to class probability first,
    # then choose the highest P(CONSISTENT).
    # --------------------------------------------------------

    chunk_probabilities = (
        torch.softmax(
            logits_float,
            dim=-1,
        )[
            :,
            1
        ]
    )

    probability_max_chunk_probability = float(
        torch.max(
            chunk_probabilities
        )
        .cpu()
        .item()
    )

    # --------------------------------------------------------
    # Variant D
    #
    # Mean chunk probability.
    # Diagnostic only.
    # --------------------------------------------------------

    probability_mean_chunk_probability = float(
        torch.mean(
            chunk_probabilities
        )
        .cpu()
        .item()
    )

    # --------------------------------------------------------
    # Variant E
    #
    # Mean logits followed by softmax.
    # Diagnostic only.
    # --------------------------------------------------------

    bag_logits_mean = torch.mean(
        logits_float,
        dim=0,
    )

    probability_mean_logits = float(
        torch.softmax(
            bag_logits_mean,
            dim=-1,
        )[
            1
        ]
        .cpu()
        .item()
    )

    return {
        "prob_current_max_logit_difference":
            probability_current,

        "prob_elementwise_max_logits":
            probability_elementwise_max_logits,

        "prob_max_chunk_probability":
            probability_max_chunk_probability,

        "prob_mean_chunk_probability":
            probability_mean_chunk_probability,

        "prob_mean_logits":
            probability_mean_logits,
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
        "INTEGRATION STEP 04"
    )
    print(
        "DIAGNOSE MAX-MIL AGGREGATION"
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
        f"Tolerance: "
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
    # LOAD REGRESSION SAMPLE
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
        "new_prob_mil",
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
    # RUN VARIANTS
    # ========================================================

    records = []

    print()
    print("=" * 80)
    print(
        "RUN AGGREGATION DIAGNOSTIC"
    )
    print("=" * 80)

    for index, row in enumerate(
        frame.itertuples(
            index=False
        ),
        start=1,
    ):

        chunk_result = (
            get_chunk_logits(
                pipeline=pipeline,
                title=(
                    row.headline_title
                ),
                content=(
                    row.anchor_content
                ),
            )
        )

        variants = compute_variants(
            chunk_result[
                "logits"
            ]
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

            "previous_pipeline_prob_mil":
                float(
                    row.new_prob_mil
                ),

            "original_article_token_count":
                int(
                    row.original_article_token_count
                ),

            "new_article_token_count":
                int(
                    chunk_result[
                        "article_token_count"
                    ]
                ),

            "original_chunk_count":
                int(
                    row.original_chunk_count
                ),

            "new_chunk_count":
                int(
                    chunk_result[
                        "chunk_count"
                    ]
                ),

            "chunk_starts":
                json.dumps(
                    chunk_result[
                        "chunk_starts"
                    ]
                ),
        }

        record.update(
            variants
        )

        for key, value in variants.items():

            record[
                key.replace(
                    "prob_",
                    "abs_error_",
                    1,
                )
            ] = abs(
                float(
                    value
                )
                -
                float(
                    row.prob_mil
                )
            )

        records.append(
            record
        )

        print(
            f"  [{index:02d}/{len(frame):02d}] "
            f"{row.pair_id} | "
            f"saved={row.prob_mil:.6f} | "
            f"current="
            f"{variants['prob_current_max_logit_difference']:.6f} | "
            f"elementwise="
            f"{variants['prob_elementwise_max_logits']:.6f}"
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
    # VERIFY RECOMPUTED CURRENT VARIANT
    # ========================================================

    current_recompute_error = np.abs(
        detail[
            "prob_current_max_logit_difference"
        ].to_numpy(
            dtype=np.float64
        )
        -
        detail[
            "previous_pipeline_prob_mil"
        ].to_numpy(
            dtype=np.float64
        )
    )

    print()
    print("=" * 80)
    print(
        "CURRENT PIPELINE RECOMPUTE CHECK"
    )
    print("=" * 80)

    print()
    print(
        f"Max difference from previous "
        f"regression run: "
        f"{current_recompute_error.max():.9f}"
    )

    # ========================================================
    # SUMMARIZE VARIANTS
    # ========================================================

    variants = {
        "current_max_logit_difference":
            "prob_current_max_logit_difference",

        "elementwise_max_logits":
            "prob_elementwise_max_logits",

        "max_chunk_probability":
            "prob_max_chunk_probability",

        "mean_chunk_probability":
            "prob_mean_chunk_probability",

        "mean_logits":
            "prob_mean_logits",
    }

    summary = {}

    print()
    print("=" * 80)
    print(
        "AGGREGATION RESULTS"
    )
    print("=" * 80)

    for name, column in variants.items():

        stats = error_stats(
            saved=(
                detail[
                    "saved_prob_mil"
                ]
            ),
            predicted=(
                detail[
                    column
                ]
            ),
        )

        summary[
            name
        ] = stats

        print()
        print(
            name
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
    # RANK VARIANTS
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
        "VARIANT RANKING"
    )
    print("=" * 80)

    print()

    for rank, (
        name,
        stats,
    ) in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{name:32s} "
            f"MAE="
            f"{stats['mean_abs_error']:.9f} "
            f"MAX="
            f"{stats['max_abs_error']:.9f}"
        )

    best_name = (
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
    # WORST CASES FOR BEST VARIANT
    # ========================================================

    best_column = variants[
        best_name
    ]

    best_error_column = (
        best_column.replace(
            "prob_",
            "abs_error_",
            1,
        )
    )

    worst = (
        detail
        .sort_values(
            best_error_column,
            ascending=False,
        )
        .head(8)
    )

    print()
    print("=" * 80)
    print(
        "WORST CASES FOR BEST VARIANT"
    )
    print("=" * 80)

    print()

    for row in worst.itertuples(
        index=False
    ):

        predicted_value = getattr(
            row,
            best_column,
        )

        error_value = getattr(
            row,
            best_error_column,
        )

        print(
            f"{row.pair_id} | "
            f"saved={row.saved_prob_mil:.6f} | "
            f"candidate={predicted_value:.6f} | "
            f"error={error_value:.6f} | "
            f"chunks={row.original_chunk_count}"
        )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    aggregation_only_fix = bool(
        best_stats[
            "pass"
        ]
    )

    report = {
        "stage":
            "integration_step_04_mil_aggregation_diagnostic",

        "validation_only":
            True,

        "training_performed":
            False,

        "test_used":
            False,

        "tolerance":
            TOLERANCE,

        "samples":
            int(
                len(
                    detail
                )
            ),

        "current_pipeline_recompute_max_error":
            float(
                current_recompute_error.max()
            ),

        "variants":
            summary,

        "ranking":
            [
                {
                    "rank":
                        int(
                            rank
                        ),

                    "variant":
                        name,

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
                    name,
                    stats,
                )
                in enumerate(
                    ranking,
                    start=1,
                )
            ],

        "best_variant":
            best_name,

        "aggregation_only_fix_supported":
            aggregation_only_fix,
    }

    save_json(
        report,
        REPORT_PATH,
    )

    print()
    print("=" * 80)
    print(
        "DIAGNOSIS"
    )
    print("=" * 80)

    print()
    print(
        f"Best variant: "
        f"{best_name}"
    )

    print(
        f"Best mean error: "
        f"{best_stats['mean_abs_error']:.9f}"
    )

    print(
        f"Best max error : "
        f"{best_stats['max_abs_error']:.9f}"
    )

    if aggregation_only_fix:

        print()
        print(
            "AGGREGATION-ONLY FIX IS SUPPORTED."
        )

        print(
            "Current chunks appear sufficient to "
            "reproduce the original MAX-MIL output."
        )

    else:

        print()
        print(
            "NO AGGREGATION VARIANT PASSED."
        )

        print(
            "The next diagnostic must inspect "
            "original chunk positions / pair construction."
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
        "INTEGRATION STEP 04 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()