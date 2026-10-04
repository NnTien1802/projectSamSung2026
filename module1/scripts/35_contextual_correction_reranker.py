from pathlib import Path
from contextlib import nullcontext
import csv
import json
import math
import time
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_wordloc_v1"
    / "validation.jsonl"
)

LOOKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_correction_v1"
    / "correction_lookup.json"
)

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_word_detector_v1"
    / "tokenizer"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_correction_reranker_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "validation_reranking_summary.json"
)

ERROR_ANALYSIS_PATH = (
    OUTPUT_DIR
    / "validation_reranking_errors.csv"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "vinai/phobert-base-v2"

MAX_LENGTH = 256

CONTEXT_WINDOW = 8

MLM_BATCH_SIZE = 24


# ============================================================
# LAMBDA SWEEP
#
# final_score =
#
# log(lookup_probability)
# +
# lambda * mlm_context_score
#
# lambda = 0 reproduces lookup ranking.
# ============================================================

LAMBDA_VALUES = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision(
        "high"
    )


# ============================================================
# NORMALIZATION
# ============================================================

def nfc(text):

    return unicodedata.normalize(
        "NFC",
        str(text),
    )


def canonical(text):

    return (
        nfc(text)
        .casefold()
    )


# ============================================================
# CASE RESTORE
# ============================================================

def restore_case(
    source,
    candidate,
):

    if not source:

        return candidate

    if source.isupper():

        return candidate.upper()

    if (
        source[0].isupper()
        and
        source[1:].islower()
    ):

        return (
            candidate[:1].upper()
            + candidate[1:]
        )

    return candidate


# ============================================================
# PUNCTUATION SPLIT
# ============================================================

def is_core_character(char):

    category = (
        unicodedata.category(
            char
        )
    )

    return (
        category.startswith("L")
        or
        category.startswith("M")
        or
        category.startswith("N")
    )


def split_affixes(text):

    text = nfc(
        text
    )

    if not text:

        return (
            "",
            "",
            "",
        )

    start = 0

    while (
        start < len(text)
        and
        not is_core_character(
            text[start]
        )
    ):

        start += 1

    end = len(
        text
    )

    while (
        end > start
        and
        not is_core_character(
            text[
                end - 1
            ]
        )
    ):

        end -= 1

    return (
        text[:start],
        text[start:end],
        text[end:],
    )


# ============================================================
# COMPONENT CANDIDATES
# ============================================================

def get_component_candidates(
    noisy_word,
    component_lookup,
):

    parts = nfc(
        noisy_word
    ).split(
        "_"
    )

    candidates = {}

    for component_index, part in enumerate(
        parts
    ):

        (
            prefix,
            core,
            suffix,
        ) = split_affixes(
            part
        )

        if not core:

            continue

        key = canonical(
            core
        )

        entries = (
            component_lookup.get(
                key
            )
        )

        if not entries:

            continue

        for entry in entries:

            corrected_core = restore_case(
                core,
                entry[
                    "candidate"
                ],
            )

            corrected_part = (
                prefix
                + corrected_core
                + suffix
            )

            corrected_parts = list(
                parts
            )

            corrected_parts[
                component_index
            ] = corrected_part

            candidate = "_".join(
                corrected_parts
            )

            if (
                canonical(
                    candidate
                )
                ==
                canonical(
                    noisy_word
                )
            ):

                continue

            value = {
                "candidate":
                    nfc(
                        candidate
                    ),

                "method":
                    "component",

                "component_index":
                    component_index,

                "support":
                    int(
                        entry[
                            "count"
                        ]
                    ),

                "conditional_probability":
                    float(
                        entry[
                            "probability"
                        ]
                    ),
            }

            old = candidates.get(
                candidate
            )

            if old is None:

                candidates[
                    candidate
                ] = value

            else:

                if (
                    value[
                        "conditional_probability"
                    ]
                    >
                    old[
                        "conditional_probability"
                    ]
                ):

                    candidates[
                        candidate
                    ] = value

                elif (
                    value[
                        "conditional_probability"
                    ]
                    ==
                    old[
                        "conditional_probability"
                    ]
                    and
                    value[
                        "support"
                    ]
                    >
                    old[
                        "support"
                    ]
                ):

                    candidates[
                        candidate
                    ] = value

    result = list(
        candidates.values()
    )

    result.sort(
        key=lambda item: (
            item[
                "conditional_probability"
            ],
            item[
                "support"
            ],
        ),
        reverse=True,
    )

    return result[:5]


# ============================================================
# CORRECTION CANDIDATES
# ============================================================

def get_correction_candidates(
    noisy_word,
    exact_lookup,
    component_lookup,
):

    noisy_word = nfc(
        noisy_word
    )

    key = canonical(
        noisy_word
    )

    entries = exact_lookup.get(
        key
    )

    # ========================================================
    # EXACT LOOKUP
    # ========================================================

    if entries:

        candidates = []

        for entry in entries:

            candidate = restore_case(
                noisy_word,
                entry[
                    "candidate"
                ],
            )

            candidates.append({
                "candidate":
                    candidate,

                "method":
                    "exact",

                "component_index":
                    None,

                "support":
                    int(
                        entry[
                            "count"
                        ]
                    ),

                "conditional_probability":
                    float(
                        entry[
                            "probability"
                        ]
                    ),
            })

        return candidates

    # ========================================================
    # COMPONENT FALLBACK
    # ========================================================

    return get_component_candidates(
        noisy_word=
            noisy_word,

        component_lookup=
            component_lookup,
    )


# ============================================================
# LOCAL CONTEXT
# ============================================================

def get_local_context(
    words,
    word_fields,
    target_index,
    window,
):

    target_field = (
        word_fields[
            target_index
        ]
    )

    left = target_index

    count = 0

    while (
        left > 0
        and
        count < window
    ):

        previous_index = (
            left - 1
        )

        if (
            word_fields[
                previous_index
            ]
            != target_field
        ):

            break

        left -= 1
        count += 1

    right = (
        target_index + 1
    )

    count = 0

    while (
        right
        < len(
            words
        )
        and
        count < window
    ):

        if (
            word_fields[
                right
            ]
            != target_field
        ):

            break

        right += 1
        count += 1

    context_words = list(
        words[
            left:right
        ]
    )

    target_position = (
        target_index
        - left
    )

    return (
        context_words,
        target_position,
    )


# ============================================================
# LOAD LOOKUP
# ============================================================

def load_lookup():

    print(
        f"\nLoading correction lookup:\n"
        f"{LOOKUP_PATH}"
    )

    with open(
        LOOKUP_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return (
        data[
            "exact_lookup"
        ],
        data[
            "component_lookup"
        ],
    )


# ============================================================
# LOAD VALIDATION CASES
# ============================================================

def load_validation_cases(
    exact_lookup,
    component_lookup,
):

    print(
        f"\nLoading validation:\n"
        f"{VALIDATION_PATH}"
    )

    start_time = time.time()

    cases = []

    document_count = 0

    with open(
        VALIDATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            row = json.loads(
                line
            )

            document_count += 1

            words = row[
                "words"
            ]

            word_fields = row[
                "word_fields"
            ]

            word_errors = row.get(
                "word_errors",
                [],
            )

            for error in word_errors:

                target_index = int(
                    error[
                        "index"
                    ]
                )

                noisy = nfc(
                    error[
                        "noisy"
                    ]
                )

                clean = nfc(
                    error[
                        "clean"
                    ]
                )

                candidates = (
                    get_correction_candidates(
                        noisy_word=
                            noisy,

                        exact_lookup=
                            exact_lookup,

                        component_lookup=
                            component_lookup,
                    )
                )

                if candidates:

                    method = (
                        candidates[
                            0
                        ][
                            "method"
                        ]
                    )

                else:

                    method = "none"

                (
                    context_words,
                    target_position,
                ) = get_local_context(
                    words=
                        words,

                    word_fields=
                        word_fields,

                    target_index=
                        target_index,

                    window=
                        CONTEXT_WINDOW,
                )

                cases.append({
                    "source_id":
                        row.get(
                            "source_id"
                        ),

                    "severity":
                        row.get(
                            "severity"
                        ),

                    "field":
                        error.get(
                            "field"
                        ),

                    "noisy":
                        noisy,

                    "clean":
                        clean,

                    "method":
                        method,

                    "candidates":
                        candidates,

                    "context_words":
                        context_words,

                    "target_position":
                        target_position,
                })

    elapsed = (
        time.time()
        - start_time
    )

    exact_count = sum(
        1
        for case in cases
        if case[
            "method"
        ]
        == "exact"
    )

    component_count = sum(
        1
        for case in cases
        if case[
            "method"
        ]
        == "component"
    )

    none_count = sum(
        1
        for case in cases
        if case[
            "method"
        ]
        == "none"
    )

    print(
        f"\nDocuments       : "
        f"{document_count:,}"
    )

    print(
        f"Error words     : "
        f"{len(cases):,}"
    )

    print(
        f"Exact cases     : "
        f"{exact_count:,}"
    )

    print(
        f"Component cases : "
        f"{component_count:,}"
    )

    print(
        f"No candidate    : "
        f"{none_count:,}"
    )

    print(
        f"Load time       : "
        f"{elapsed:.2f}s"
    )

    return cases


# ============================================================
# FIND CONTENT OFFSET
# ============================================================

def find_content_offset(
    full_ids,
    content_ids,
):

    if not content_ids:

        return 0

    max_start = (
        len(
            full_ids
        )
        - len(
            content_ids
        )
    )

    for start in range(
        max_start + 1
    ):

        if (
            full_ids[
                start:
                start
                + len(
                    content_ids
                )
            ]
            == content_ids
        ):

            return start

    raise RuntimeError(
        "Could not locate content "
        "inside special-token sequence."
    )


# ============================================================
# BUILD MLM FEATURE
# ============================================================

def build_mlm_feature(
    prefix_ids,
    suffix_ids,
    candidate,
    tokenizer,
):

    candidate_ids = tokenizer(
        candidate,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )[
        "input_ids"
    ]

    if not candidate_ids:

        return None

    mask_count = len(
        candidate_ids
    )

    special_tokens = (
        tokenizer
        .num_special_tokens_to_add(
            pair=False
        )
    )

    content_length = (
        MAX_LENGTH
        - special_tokens
    )

    available_context = (
        content_length
        - mask_count
    )

    if available_context < 0:

        return None

    # ========================================================
    # Keep context nearest to target.
    # ========================================================

    left_budget = (
        available_context
        // 2
    )

    right_budget = (
        available_context
        - left_budget
    )

    left_ids = (
        prefix_ids[
            -left_budget:
        ]
        if left_budget > 0
        else []
    )

    right_ids = (
        suffix_ids[
            :right_budget
        ]
        if right_budget > 0
        else []
    )

    # Use unused capacity from one side on the other side.
    unused_left = (
        left_budget
        - len(
            left_ids
        )
    )

    unused_right = (
        right_budget
        - len(
            right_ids
        )
    )

    if (
        unused_left > 0
        and
        len(
            suffix_ids
        )
        > len(
            right_ids
        )
    ):

        extra = suffix_ids[
            len(
                right_ids
            ):
            len(
                right_ids
            )
            + unused_left
        ]

        right_ids = (
            right_ids
            + extra
        )

    if (
        unused_right > 0
        and
        len(
            prefix_ids
        )
        > len(
            left_ids
        )
    ):

        desired = min(
            len(
                prefix_ids
            ),
            len(
                left_ids
            )
            + unused_right,
        )

        left_ids = prefix_ids[
            -desired:
        ]

    mask_ids = [
        tokenizer.mask_token_id
    ] * mask_count

    content_ids = (
        left_ids
        + mask_ids
        + right_ids
    )

    full_ids = (
        tokenizer
        .build_inputs_with_special_tokens(
            content_ids
        )
    )

    content_offset = (
        find_content_offset(
            full_ids,
            content_ids,
        )
    )

    mask_start = (
        content_offset
        + len(
            left_ids
        )
    )

    mask_positions = list(
        range(
            mask_start,
            mask_start
            + mask_count,
        )
    )

    return {
        "input_ids":
            full_ids,

        "attention_mask":
            [1]
            * len(
                full_ids
            ),

        "mask_positions":
            mask_positions,

        "candidate_ids":
            candidate_ids,
    }


# ============================================================
# PREPARE MLM TASKS
#
# Only component fallback cases need MLM.
# Exact lookup remains unchanged.
# ============================================================

def prepare_mlm_tasks(
    cases,
    tokenizer,
):

    print(
        "\nPreparing component reranking tasks..."
    )

    tasks = []

    component_case_count = 0

    for case_index, case in enumerate(
        tqdm(
            cases,
            desc="Preparing",
        )
    ):

        if (
            case[
                "method"
            ]
            != "component"
        ):

            continue

        component_case_count += 1

        context_words = (
            case[
                "context_words"
            ]
        )

        target_position = int(
            case[
                "target_position"
            ]
        )

        # ====================================================
        # Tokenize local context once per error.
        # ====================================================

        encoded_words = tokenizer(
            context_words,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )[
            "input_ids"
        ]

        prefix_ids = []

        for ids in encoded_words[
            :target_position
        ]:

            prefix_ids.extend(
                ids
            )

        suffix_ids = []

        for ids in encoded_words[
            target_position
            + 1:
        ]:

            suffix_ids.extend(
                ids
            )

        for candidate_index, candidate in enumerate(
            case[
                "candidates"
            ]
        ):

            feature = (
                build_mlm_feature(
                    prefix_ids=
                        prefix_ids,

                    suffix_ids=
                        suffix_ids,

                    candidate=
                        candidate[
                            "candidate"
                        ],

                    tokenizer=
                        tokenizer,
                )
            )

            if feature is None:

                candidate[
                    "mlm_score"
                ] = float(
                    "-inf"
                )

                continue

            tasks.append({
                "case_index":
                    case_index,

                "candidate_index":
                    candidate_index,

                **feature,
            })

    print(
        f"\nComponent cases : "
        f"{component_case_count:,}"
    )

    print(
        f"MLM sequences   : "
        f"{len(tasks):,}"
    )

    return tasks


# ============================================================
# AMP
# ============================================================

def autocast_context():

    if USE_AMP:

        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


# ============================================================
# SCORE TASKS
# ============================================================

def score_tasks(
    cases,
    tasks,
    tokenizer,
    model,
):

    print(
        "\nScoring candidates with PhoBERT MLM..."
    )

    start_time = time.time()

    for batch_start in tqdm(
        range(
            0,
            len(
                tasks
            ),
            MLM_BATCH_SIZE,
        ),
        desc="MLM reranking",
    ):

        batch_tasks = tasks[
            batch_start:
            batch_start
            + MLM_BATCH_SIZE
        ]

        features = [
            {
                "input_ids":
                    task[
                        "input_ids"
                    ],

                "attention_mask":
                    task[
                        "attention_mask"
                    ],
            }
            for task in batch_tasks
        ]

        padded = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        input_ids = (
            padded[
                "input_ids"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        attention_mask = (
            padded[
                "attention_mask"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        with torch.inference_mode():

            with autocast_context():

                output = model(
                    input_ids=
                        input_ids,

                    attention_mask=
                        attention_mask,
                )

                logits = (
                    output.logits
                )

        # ====================================================
        # Collect only logits at mask positions.
        #
        # Avoid computing log_softmax for all sequence tokens.
        # ====================================================

        batch_indices = []

        token_positions = []

        target_token_ids = []

        owner_indices = []

        for local_index, task in enumerate(
            batch_tasks
        ):

            mask_positions = (
                task[
                    "mask_positions"
                ]
            )

            candidate_ids = (
                task[
                    "candidate_ids"
                ]
            )

            for (
                position,
                token_id,
            ) in zip(
                mask_positions,
                candidate_ids,
            ):

                batch_indices.append(
                    local_index
                )

                token_positions.append(
                    position
                )

                target_token_ids.append(
                    token_id
                )

                owner_indices.append(
                    local_index
                )

        batch_indices_tensor = torch.tensor(
            batch_indices,
            dtype=torch.long,
            device=DEVICE,
        )

        positions_tensor = torch.tensor(
            token_positions,
            dtype=torch.long,
            device=DEVICE,
        )

        targets_tensor = torch.tensor(
            target_token_ids,
            dtype=torch.long,
            device=DEVICE,
        )

        selected_logits = logits[
            batch_indices_tensor,
            positions_tensor,
            :
        ]

        selected_log_probs = (
            F.log_softmax(
                selected_logits.float(),
                dim=-1,
            )
        )

        token_scores = (
            selected_log_probs[
                torch.arange(
                    len(
                        targets_tensor
                    ),
                    device=DEVICE,
                ),
                targets_tensor,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # Mean log probability per candidate.
        #
        # Mean avoids automatically penalizing candidates
        # that tokenize into more subwords.
        # ====================================================

        candidate_scores = [
            []
            for _ in batch_tasks
        ]

        for (
            owner,
            score,
        ) in zip(
            owner_indices,
            token_scores,
        ):

            candidate_scores[
                owner
            ].append(
                float(
                    score
                )
            )

        for local_index, task in enumerate(
            batch_tasks
        ):

            scores = candidate_scores[
                local_index
            ]

            if scores:

                mlm_score = float(
                    np.mean(
                        scores
                    )
                )

            else:

                mlm_score = float(
                    "-inf"
                )

            case = cases[
                task[
                    "case_index"
                ]
            ]

            candidate = case[
                "candidates"
            ][
                task[
                    "candidate_index"
                ]
            ]

            candidate[
                "mlm_score"
            ] = mlm_score

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nMLM scoring time: "
        f"{elapsed:.2f}s"
    )

    return elapsed


# ============================================================
# RANK CANDIDATES
# ============================================================

def rank_case(
    case,
    lambda_value,
    mlm_only=False,
):

    candidates = case[
        "candidates"
    ]

    if not candidates:

        return []

    # Exact lookup is deliberately NOT reranked.
    if (
        case[
            "method"
        ]
        == "exact"
    ):

        return candidates

    scored = []

    for original_index, candidate in enumerate(
        candidates
    ):

        probability = max(
            float(
                candidate[
                    "conditional_probability"
                ]
            ),
            1e-12,
        )

        base_score = math.log(
            probability
        )

        mlm_score = float(
            candidate.get(
                "mlm_score",
                float(
                    "-inf"
                ),
            )
        )

        if not np.isfinite(
            mlm_score
        ):

            final_score = float(
                "-inf"
            )

        elif mlm_only:

            final_score = (
                mlm_score
            )

        else:

            final_score = (
                base_score
                +
                lambda_value
                * mlm_score
            )

        scored.append(
            (
                final_score,
                original_index,
                candidate,
            )
        )

    # Python sort is stable.
    # original_index protects baseline tie ordering.
    scored.sort(
        key=lambda item: (
            item[
                0
            ],
            -item[
                1
            ],
        ),
        reverse=True,
    )

    return [
        item[
            2
        ]
        for item in scored
    ]


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    cases,
    lambda_value,
    mlm_only=False,
):

    total = len(
        cases
    )

    covered = 0

    top1 = 0
    top3 = 0
    top5 = 0

    exact_total = 0
    exact_top1 = 0

    component_total = 0
    component_top1 = 0

    component_top3 = 0
    component_top5 = 0

    changed_component_top1 = 0

    for case in cases:

        candidates = (
            rank_case(
                case=
                    case,

                lambda_value=
                    lambda_value,

                mlm_only=
                    mlm_only,
            )
        )

        if not candidates:

            continue

        covered += 1

        gold = nfc(
            case[
                "clean"
            ]
        )

        predictions = [
            nfc(
                candidate[
                    "candidate"
                ]
            )
            for candidate
            in candidates
        ]

        top1_correct = (
            predictions[
                0
            ]
            == gold
        )

        if top1_correct:

            top1 += 1

        if (
            gold
            in predictions[
                :3
            ]
        ):

            top3 += 1

        if (
            gold
            in predictions[
                :5
            ]
        ):

            top5 += 1

        if (
            case[
                "method"
            ]
            == "exact"
        ):

            exact_total += 1

            if top1_correct:

                exact_top1 += 1

        elif (
            case[
                "method"
            ]
            == "component"
        ):

            component_total += 1

            if top1_correct:

                component_top1 += 1

            if (
                gold
                in predictions[
                    :3
                ]
            ):

                component_top3 += 1

            if (
                gold
                in predictions[
                    :5
                ]
            ):

                component_top5 += 1

            original_prediction = nfc(
                case[
                    "candidates"
                ][
                    0
                ][
                    "candidate"
                ]
            )

            if (
                predictions[
                    0
                ]
                != original_prediction
            ):

                changed_component_top1 += 1

    return {
        "lambda":
            (
                "mlm_only"
                if mlm_only
                else float(
                    lambda_value
                )
            ),

        "total":
            total,

        "covered":
            covered,

        "coverage":
            (
                covered / total
                if total
                else 0.0
            ),

        "top1":
            top1,

        "top1_accuracy":
            (
                top1 / total
                if total
                else 0.0
            ),

        "top3_accuracy":
            (
                top3 / total
                if total
                else 0.0
            ),

        "top5_accuracy":
            (
                top5 / total
                if total
                else 0.0
            ),

        "top1_when_covered":
            (
                top1 / covered
                if covered
                else 0.0
            ),

        "exact_total":
            exact_total,

        "exact_top1_accuracy":
            (
                exact_top1
                / exact_total
                if exact_total
                else 0.0
            ),

        "component_total":
            component_total,

        "component_top1_accuracy":
            (
                component_top1
                / component_total
                if component_total
                else 0.0
            ),

        "component_top3_accuracy":
            (
                component_top3
                / component_total
                if component_total
                else 0.0
            ),

        "component_top5_accuracy":
            (
                component_top5
                / component_total
                if component_total
                else 0.0
            ),

        "component_top1_changed":
            changed_component_top1,
    }


# ============================================================
# ERROR ANALYSIS
# ============================================================

def save_error_analysis(
    cases,
    best_lambda,
):

    rows = []

    for case in cases:

        candidates = rank_case(
            case=
                case,

            lambda_value=
                best_lambda,
        )

        if candidates:

            prediction = (
                candidates[
                    0
                ][
                    "candidate"
                ]
            )

            mlm_score = (
                candidates[
                    0
                ].get(
                    "mlm_score"
                )
            )

        else:

            prediction = ""

            mlm_score = None

        correct = (
            nfc(
                prediction
            )
            ==
            nfc(
                case[
                    "clean"
                ]
            )
        )

        if correct:

            continue

        context = " ".join(
            case[
                "context_words"
            ]
        )

        rows.append({
            "source_id":
                case[
                    "source_id"
                ],

            "severity":
                case[
                    "severity"
                ],

            "field":
                case[
                    "field"
                ],

            "method":
                case[
                    "method"
                ],

            "noisy":
                case[
                    "noisy"
                ],

            "clean":
                case[
                    "clean"
                ],

            "prediction":
                prediction,

            "prediction_mlm_score":
                (
                    ""
                    if mlm_score is None
                    else mlm_score
                ),

            "context":
                context,
        })

    with open(
        ERROR_ANALYSIS_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "source_id",
            "severity",
            "field",
            "method",
            "noisy",
            "clean",
            "prediction",
            "prediction_mlm_score",
            "context",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print(
        "\n" + "-" * 80
    )

    print(
        title
    )

    print(
        "-" * 80
    )

    print(
        f"Top-1 overall          : "
        f"{metrics['top1_accuracy']:.6f}"
    )

    print(
        f"Top-3 overall          : "
        f"{metrics['top3_accuracy']:.6f}"
    )

    print(
        f"Top-5 overall          : "
        f"{metrics['top5_accuracy']:.6f}"
    )

    print(
        f"Top-1 when covered     : "
        f"{metrics['top1_when_covered']:.6f}"
    )

    print(
        f"Exact Top-1            : "
        f"{metrics['exact_top1_accuracy']:.6f}"
    )

    print(
        f"Component Top-1        : "
        f"{metrics['component_top1_accuracy']:.6f}"
    )

    print(
        f"Component Top-3        : "
        f"{metrics['component_top3_accuracy']:.6f}"
    )

    print(
        f"Component Top-5        : "
        f"{metrics['component_top5_accuracy']:.6f}"
    )

    print(
        f"Component rank changed : "
        f"{metrics['component_top1_changed']:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "VN-NEWSSPELL - CONTEXTUAL CORRECTION RERANKER"
    )

    print("=" * 80)

    print(
        "\nCandidate lookup: TRAIN only."
    )

    print(
        "Reranker tuning : VALIDATION only."
    )

    print(
        "Test set is NOT read or used."
    )

    print(
        "\nStrategy:"
    )

    print(
        "  exact lookup     -> unchanged"
    )

    print(
        "  component lookup -> PhoBERT MLM reranking"
    )

    print(
        f"\nDevice: "
        f"{DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # LOOKUP
    # ========================================================

    (
        exact_lookup,
        component_lookup,
    ) = load_lookup()

    # ========================================================
    # VALIDATION CASES
    # ========================================================

    cases = load_validation_cases(
        exact_lookup=
            exact_lookup,

        component_lookup=
            component_lookup,
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\nLoading PhoBERT tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_DIR,
            use_fast=False,
        )
    )

    if tokenizer.mask_token_id is None:

        raise RuntimeError(
            "Tokenizer has no mask token."
        )

    print(
        f"Mask token    : "
        f"{tokenizer.mask_token}"
    )

    print(
        f"Mask token ID : "
        f"{tokenizer.mask_token_id}"
    )

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = evaluate(
        cases=
            cases,

        lambda_value=
            0.0,
    )

    print_metrics(
        "BASELINE LOOKUP - LAMBDA = 0",
        baseline,
    )

    # ========================================================
    # MLM TASKS
    # ========================================================

    tasks = prepare_mlm_tasks(
        cases=
            cases,

        tokenizer=
            tokenizer,
    )

    # ========================================================
    # MLM MODEL
    # ========================================================

    print(
        f"\nLoading PhoBERT Masked LM:\n"
        f"{MODEL_NAME}"
    )

    model = (
        AutoModelForMaskedLM
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    # ========================================================
    # SCORE
    # ========================================================

    mlm_seconds = score_tasks(
        cases=
            cases,

        tasks=
            tasks,

        tokenizer=
            tokenizer,

        model=
            model,
    )

    # ========================================================
    # RELEASE GPU
    # ========================================================

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # LAMBDA SWEEP
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION LAMBDA SWEEP"
    )

    print(
        "=" * 80
    )

    sweep = []

    for lambda_value in (
        LAMBDA_VALUES
    ):

        metrics = evaluate(
            cases=
                cases,

            lambda_value=
                lambda_value,
        )

        sweep.append(
            metrics
        )

        print(
            f"\nlambda={lambda_value:>4.2f}"
            f" | overall Top1="
            f"{metrics['top1_accuracy']:.6f}"
            f" | component Top1="
            f"{metrics['component_top1_accuracy']:.6f}"
            f" | component Top3="
            f"{metrics['component_top3_accuracy']:.6f}"
        )

    # ========================================================
    # MLM ONLY
    # ========================================================

    mlm_only = evaluate(
        cases=
            cases,

        lambda_value=
            0.0,

        mlm_only=
            True,
    )

    print(
        f"\nMLM only"
        f"    | overall Top1="
        f"{mlm_only['top1_accuracy']:.6f}"
        f" | component Top1="
        f"{mlm_only['component_top1_accuracy']:.6f}"
    )

    # ========================================================
    # SELECT BEST
    #
    # Primary metric:
    # overall validation Top-1 correction accuracy.
    #
    # Exact cases are unchanged, so improvement must come
    # from component fallback.
    # ========================================================

    best = max(
        sweep,
        key=lambda item: (
            item[
                "top1_accuracy"
            ],
            item[
                "component_top1_accuracy"
            ],
            item[
                "top3_accuracy"
            ],
            -float(
                item[
                    "lambda"
                ]
            ),
        ),
    )

    best_lambda = float(
        best[
            "lambda"
        ]
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "SELECTED CONTEXTUAL RERANKER"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSelected lambda: "
        f"{best_lambda:.4f}"
    )

    print_metrics(
        "BEST VALIDATION RESULT",
        best,
    )

    # ========================================================
    # DELTA
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "BASELINE vs CONTEXTUAL RERANKER"
    )

    print(
        "=" * 80
    )

    print(
        f"\nOverall Top-1:"
    )

    print(
        f"  baseline : "
        f"{baseline['top1_accuracy']:.6f}"
    )

    print(
        f"  reranked : "
        f"{best['top1_accuracy']:.6f}"
    )

    print(
        f"  delta    : "
        f"{best['top1_accuracy'] - baseline['top1_accuracy']:+.6f}"
    )

    print(
        f"\nComponent Top-1:"
    )

    print(
        f"  baseline : "
        f"{baseline['component_top1_accuracy']:.6f}"
    )

    print(
        f"  reranked : "
        f"{best['component_top1_accuracy']:.6f}"
    )

    print(
        f"  delta    : "
        f"{best['component_top1_accuracy'] - baseline['component_top1_accuracy']:+.6f}"
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    save_error_analysis(
        cases=
            cases,

        best_lambda=
            best_lambda,
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "dataset":
            "validation",

        "test_used":
            False,

        "candidate_source":
            "train_only",

        "reranker":
            "vinai/phobert-base-v2 masked language model",

        "reranked_methods": [
            "component",
        ],

        "exact_lookup_reranked":
            False,

        "context_window_words_each_side":
            CONTEXT_WINDOW,

        "max_length":
            MAX_LENGTH,

        "mlm_batch_size":
            MLM_BATCH_SIZE,

        "lambda_values":
            LAMBDA_VALUES,

        "baseline":
            baseline,

        "sweep":
            sweep,

        "mlm_only":
            mlm_only,

        "selected_lambda":
            best_lambda,

        "selected_metrics":
            best,

        "mlm_scoring_seconds":
            mlm_seconds,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CONTEXTUAL RERANKING FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nError analysis:\n"
        f"{ERROR_ANALYSIS_PATH}"
    )

    print(
        "\nTest set was NOT read or used."
    )


if __name__ == "__main__":

    main()