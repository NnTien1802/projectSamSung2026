from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import csv
import json

import numpy as np
import torch

from transformers import AutoTokenizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_v1"
    / "validation.jsonl"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_all_fields"
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

TOKENIZER_DIR = (
    MODEL_DIR
    / "tokenizer"
)

THRESHOLD_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_all_fields"
    / "threshold_tuning"
)

PREDICTIONS_PATH = (
    THRESHOLD_DIR
    / "validation_probabilities.csv"
)

BEST_THRESHOLD_PATH = (
    THRESHOLD_DIR
    / "best_threshold.json"
)

OUTPUT_PATH = (
    THRESHOLD_DIR
    / "validation_chunk_coverage_analysis.json"
)

DETAIL_CSV_PATH = (
    THRESHOLD_DIR
    / "validation_chunk_coverage_details.csv"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    try:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

    required = [
        "text_fields",
        "max_length",
        "stride",
        "max_chunks_per_document",
    ]

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:

        raise RuntimeError(
            "Checkpoint missing keys: "
            + ", ".join(missing)
        )

    return checkpoint


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    with open(
        BEST_THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return float(
        data["selected_threshold"]
    )


# ============================================================
# BUILD DOCUMENT TEXT
# ============================================================

def build_document_text(
    row,
    text_fields,
    field_separator,
):

    parts = []

    for field in text_fields:

        text = row.get(
            field,
            "",
        )

        if text is None:
            text = ""

        text = str(
            text
        ).strip()

        if text:

            parts.append(
                text
            )

    return field_separator.join(
        parts
    )


# ============================================================
# LOAD VALIDATION AS CLEAN/NOISY PAIRS
# ============================================================

def load_validation_pairs():

    groups = {}

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

            source_id = row[
                "source_id"
            ]

            label = int(
                row[
                    "label"
                ]
            )

            if source_id not in groups:

                groups[source_id] = {}

            if label == 0:

                groups[source_id][
                    "clean"
                ] = row

            elif label == 1:

                groups[source_id][
                    "noisy"
                ] = row

    invalid = []

    for source_id, group in (
        groups.items()
    ):

        if (
            "clean" not in group
            or
            "noisy" not in group
        ):

            invalid.append(
                source_id
            )

    if invalid:

        raise RuntimeError(
            f"Invalid source groups: "
            f"{len(invalid):,}"
        )

    return groups


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

def load_predictions():

    predictions = {}

    with open(
        PREDICTIONS_PATH,
        "r",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if int(
                row["label"]
            ) != 1:

                continue

            predictions[
                row["variant_id"]
            ] = float(
                row[
                    "probability_error"
                ]
            )

    return predictions


# ============================================================
# CHUNK SELECTION
# ============================================================

def evenly_spaced_indices(
    total_chunks,
    max_chunks,
):

    if total_chunks <= max_chunks:

        return list(
            range(
                total_chunks
            )
        )

    indices = np.linspace(
        0,
        total_chunks - 1,
        max_chunks,
        dtype=int,
    )

    return sorted(
        set(
            indices.tolist()
        )
    )


# ============================================================
# ALL / SELECTED CHUNK RANGES
# ============================================================

def get_chunk_ranges(
    total_tokens,
    content_length,
    stride,
    max_chunks,
):

    if total_tokens == 0:

        all_ranges = [
            {
                "chunk_index": 0,
                "start_token": 0,
                "end_token": 0,
            }
        ]

        return (
            all_ranges,
            all_ranges,
        )

    step_size = (
        content_length
        - stride
    )

    starts = []

    start = 0

    while True:

        starts.append(
            start
        )

        if (
            start
            + content_length
            >= total_tokens
        ):

            break

        start += (
            step_size
        )

    all_ranges = []

    for chunk_index, start in enumerate(
        starts
    ):

        all_ranges.append({
            "chunk_index":
                int(
                    chunk_index
                ),

            "start_token":
                int(
                    start
                ),

            "end_token":
                int(
                    min(
                        start
                        + content_length,
                        total_tokens,
                    )
                ),
        })

    selected_indices = (
        evenly_spaced_indices(
            len(
                all_ranges
            ),
            max_chunks,
        )
    )

    selected_ranges = [
        all_ranges[index]
        for index
        in selected_indices
    ]

    return (
        all_ranges,
        selected_ranges,
    )


# ============================================================
# TOKEN SEQUENCE DIFFERENCES
# ============================================================

def get_token_differences(
    clean_ids,
    noisy_ids,
):

    """
    Compare exactly what PhoBERT receives.

    a = clean token sequence
    b = noisy token sequence

    Returned spans are expressed in NOISY token coordinates.
    """

    matcher = SequenceMatcher(
        a=clean_ids,
        b=noisy_ids,
        autojunk=False,
    )

    changes = []

    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in matcher.get_opcodes():

        if tag == "equal":
            continue

        changes.append({
            "tag":
                tag,

            "clean_start":
                int(
                    i1
                ),

            "clean_end":
                int(
                    i2
                ),

            "noisy_start":
                int(
                    j1
                ),

            "noisy_end":
                int(
                    j2
                ),
        })

    return changes


# ============================================================
# CHANGE VISIBILITY
# ============================================================

def change_visible(
    change,
    selected_ranges,
    noisy_token_count,
):

    start = change[
        "noisy_start"
    ]

    end = change[
        "noisy_end"
    ]

    # --------------------------------------------------------
    # Normal replace/insert:
    # changed noisy tokens exist.
    # --------------------------------------------------------

    if end > start:

        for chunk in selected_ranges:

            chunk_start = chunk[
                "start_token"
            ]

            chunk_end = chunk[
                "end_token"
            ]

            if (
                start < chunk_end
                and
                end > chunk_start
            ):

                return True

        return False

    # --------------------------------------------------------
    # Token-level deletion:
    #
    # noisy span has zero length because something present
    # in clean disappeared in noisy.
    #
    # Model can observe this only if it sees the boundary
    # around the deletion.
    # --------------------------------------------------------

    if noisy_token_count == 0:

        return True

    boundary_start = max(
        0,
        start - 1,
    )

    boundary_end = min(
        noisy_token_count,
        start + 1,
    )

    if boundary_end <= boundary_start:

        boundary_end = min(
            noisy_token_count,
            boundary_start + 1,
        )

    for chunk in selected_ranges:

        chunk_start = chunk[
            "start_token"
        ]

        chunk_end = chunk[
            "end_token"
        ]

        if (
            boundary_start < chunk_end
            and
            boundary_end > chunk_start
        ):

            return True

    return False


# ============================================================
# PERCENTAGE
# ============================================================

def percentage(
    value,
    total,
):

    if total == 0:

        return 0.0

    return (
        value
        / total
        * 100.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "PHOBERT MIL ALL-FIELDS - "
        "VALIDATION TOKEN-DIFF CHUNK COVERAGE"
    )

    print("=" * 80)

    print(
        "\nValidation set ONLY."
    )

    print(
        "Test set is NOT read or used."
    )

    # ========================================================
    # CONFIG
    # ========================================================

    checkpoint = (
        load_checkpoint()
    )

    threshold = (
        load_threshold()
    )

    text_fields = checkpoint[
        "text_fields"
    ]

    field_separator = (
        checkpoint.get(
            "field_separator",
            " . ",
        )
    )

    max_length = int(
        checkpoint[
            "max_length"
        ]
    )

    stride = int(
        checkpoint[
            "stride"
        ]
    )

    max_chunks = int(
        checkpoint[
            "max_chunks_per_document"
        ]
    )

    print(
        f"\nLocked threshold : "
        f"{threshold:.4f}"
    )

    print(
        f"MAX_LENGTH       : "
        f"{max_length}"
    )

    print(
        f"STRIDE           : "
        f"{stride}"
    )

    print(
        f"MAX_CHUNKS       : "
        f"{max_chunks}"
    )

    print(
        f"Field separator  : "
        f"{repr(field_separator)}"
    )

    print(
        "\nText fields:"
    )

    for field in text_fields:

        print(
            f"  {field}"
        )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_DIR,
            use_fast=False,
        )
    )

    print(
        f"Tokenizer fast: "
        f"{tokenizer.is_fast}"
    )

    special_tokens = (
        tokenizer
        .num_special_tokens_to_add(
            pair=False
        )
    )

    content_length = (
        max_length
        - special_tokens
    )

    step_size = (
        content_length
        - stride
    )

    print(
        f"Special tokens   : "
        f"{special_tokens}"
    )

    print(
        f"Content length   : "
        f"{content_length}"
    )

    print(
        f"Chunk step       : "
        f"{step_size}"
    )

    # ========================================================
    # DATA
    # ========================================================

    print(
        "\nLoading validation pairs..."
    )

    groups = (
        load_validation_pairs()
    )

    predictions = (
        load_predictions()
    )

    print(
        f"Source pairs      : "
        f"{len(groups):,}"
    )

    print(
        f"Noisy predictions : "
        f"{len(predictions):,}"
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    total_noisy = 0

    total_tp = 0
    total_fn = 0

    # At document level
    any_change_visible = 0
    no_change_visible = 0

    tp_visible = 0
    tp_not_visible = 0

    fn_visible = 0
    fn_not_visible = 0

    # Exact input contradiction / tokenizer normalization
    identical_clean_noisy_text = 0
    identical_clean_noisy_tokens = 0

    # Single-error docs
    single_error_total = 0
    single_error_visible = 0
    single_error_not_visible = 0

    single_error_fn = 0
    single_error_fn_visible = 0
    single_error_fn_not_visible = 0

    # Severity
    fn_visible_severity = Counter()
    fn_not_visible_severity = Counter()

    # Metadata fields/types for visible-vs-invisible FN docs.
    fn_visible_fields = Counter()
    fn_not_visible_fields = Counter()

    fn_visible_types = Counter()
    fn_not_visible_types = Counter()

    # Chunk distribution
    total_chunk_distribution = Counter()
    selected_chunk_distribution = Counter()

    # Token-diff statistics
    change_opcode_counter = Counter()

    total_change_spans = 0
    visible_change_spans = 0
    invisible_change_spans = 0

    # Potential coverage at higher chunk limits without
    # retraining. This only measures whether the corrupted
    # token region WOULD be included.
    simulated_max_chunks = [
        3,
        4,
        5,
        6,
        8,
    ]

    simulated_document_coverage = {
        value: 0
        for value in simulated_max_chunks
    }

    simulated_fn_coverage = {
        value: 0
        for value in simulated_max_chunks
    }

    detail_rows = []

    # ========================================================
    # PROCESS EACH CLEAN/NOISY PAIR
    # ========================================================

    for source_index, (
        source_id,
        group,
    ) in enumerate(
        groups.items(),
        start=1,
    ):

        clean = group[
            "clean"
        ]

        noisy = group[
            "noisy"
        ]

        variant_id = noisy[
            "variant_id"
        ]

        if variant_id not in predictions:

            raise RuntimeError(
                f"Missing prediction for "
                f"{variant_id}"
            )

        probability = (
            predictions[
                variant_id
            ]
        )

        predicted = int(
            probability
            >= threshold
        )

        is_fn = (
            predicted == 0
        )

        total_noisy += 1

        if is_fn:

            total_fn += 1

        else:

            total_tp += 1

        # ====================================================
        # BUILD EXACT TRAINING INPUT
        # ====================================================

        clean_text = (
            build_document_text(
                clean,
                text_fields,
                field_separator,
            )
        )

        noisy_text = (
            build_document_text(
                noisy,
                text_fields,
                field_separator,
            )
        )

        if clean_text == noisy_text:

            identical_clean_noisy_text += 1

        # ====================================================
        # TOKENIZE EXACTLY AS TRAINING COLLATOR
        # ====================================================

        clean_ids = tokenizer(
            clean_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )[
            "input_ids"
        ]

        noisy_ids = tokenizer(
            noisy_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )[
            "input_ids"
        ]

        if clean_ids == noisy_ids:

            identical_clean_noisy_tokens += 1

        # ====================================================
        # TOKEN DIFFERENCES
        # ====================================================

        changes = (
            get_token_differences(
                clean_ids,
                noisy_ids,
            )
        )

        for change in changes:

            change_opcode_counter[
                change[
                    "tag"
                ]
            ] += 1

        total_change_spans += len(
            changes
        )

        # ====================================================
        # ACTUAL MAX_CHUNKS = 3
        # ====================================================

        (
            all_ranges,
            selected_ranges,
        ) = get_chunk_ranges(
            total_tokens=
                len(
                    noisy_ids
                ),

            content_length=
                content_length,

            stride=
                stride,

            max_chunks=
                max_chunks,
        )

        total_chunk_distribution[
            len(
                all_ranges
            )
        ] += 1

        selected_chunk_distribution[
            len(
                selected_ranges
            )
        ] += 1

        change_visibility = []

        for change in changes:

            visible = (
                change_visible(
                    change,
                    selected_ranges,
                    len(
                        noisy_ids
                    ),
                )
            )

            change_visibility.append(
                visible
            )

            if visible:

                visible_change_spans += 1

            else:

                invisible_change_spans += 1

        document_visible = any(
            change_visibility
        )

        if document_visible:

            any_change_visible += 1

        else:

            no_change_visible += 1

        # ====================================================
        # TP / FN COVERAGE
        # ====================================================

        if is_fn:

            if document_visible:

                fn_visible += 1

                fn_visible_severity[
                    noisy.get(
                        "severity",
                        "unknown",
                    )
                ] += 1

            else:

                fn_not_visible += 1

                fn_not_visible_severity[
                    noisy.get(
                        "severity",
                        "unknown",
                    )
                ] += 1

        else:

            if document_visible:

                tp_visible += 1

            else:

                tp_not_visible += 1

        # ====================================================
        # METADATA ERROR INFO
        # ====================================================

        errors = noisy.get(
            "errors",
            [],
        )

        if errors is None:

            errors = []

        if is_fn:

            fields = set()

            types = set()

            for error in errors:

                fields.add(
                    error.get(
                        "field",
                        "unknown",
                    )
                )

                types.add(
                    error.get(
                        "error_type",
                        "unknown",
                    )
                )

            if document_visible:

                for field in fields:

                    fn_visible_fields[
                        field
                    ] += 1

                for error_type in types:

                    fn_visible_types[
                        error_type
                    ] += 1

            else:

                for field in fields:

                    fn_not_visible_fields[
                        field
                    ] += 1

                for error_type in types:

                    fn_not_visible_types[
                        error_type
                    ] += 1

        # ====================================================
        # SINGLE-ERROR DOCUMENTS
        # ====================================================

        if len(errors) == 1:

            single_error_total += 1

            if document_visible:

                single_error_visible += 1

            else:

                single_error_not_visible += 1

            if is_fn:

                single_error_fn += 1

                if document_visible:

                    single_error_fn_visible += 1

                else:

                    single_error_fn_not_visible += 1

        # ====================================================
        # SIMULATE OTHER MAX_CHUNKS VALUES
        #
        # Important:
        # this does NOT estimate future model accuracy.
        # It only answers whether the changed token span
        # would become visible.
        # ====================================================

        simulated = {}

        for candidate_max_chunks in (
            simulated_max_chunks
        ):

            (
                _,
                candidate_ranges,
            ) = get_chunk_ranges(
                total_tokens=
                    len(
                        noisy_ids
                    ),

                content_length=
                    content_length,

                stride=
                    stride,

                max_chunks=
                    candidate_max_chunks,
            )

            candidate_visible = any(
                change_visible(
                    change,
                    candidate_ranges,
                    len(
                        noisy_ids
                    ),
                )
                for change in changes
            )

            simulated[
                candidate_max_chunks
            ] = candidate_visible

            if candidate_visible:

                simulated_document_coverage[
                    candidate_max_chunks
                ] += 1

                if is_fn:

                    simulated_fn_coverage[
                        candidate_max_chunks
                    ] += 1

        # ====================================================
        # DETAILS
        # ====================================================

        detail_rows.append({
            "source_id":
                source_id,

            "variant_id":
                variant_id,

            "severity":
                noisy.get(
                    "severity",
                    "unknown",
                ),

            "error_count":
                len(errors),

            "probability_error":
                probability,

            "prediction":
                predicted,

            "is_false_negative":
                int(
                    is_fn
                ),

            "clean_tokens":
                len(
                    clean_ids
                ),

            "noisy_tokens":
                len(
                    noisy_ids
                ),

            "total_chunks":
                len(
                    all_ranges
                ),

            "selected_chunks":
                len(
                    selected_ranges
                ),

            "token_change_spans":
                len(
                    changes
                ),

            "any_change_visible":
                int(
                    document_visible
                ),

            "visible_with_3_chunks":
                int(
                    simulated[3]
                ),

            "visible_with_4_chunks":
                int(
                    simulated[4]
                ),

            "visible_with_5_chunks":
                int(
                    simulated[5]
                ),

            "visible_with_6_chunks":
                int(
                    simulated[6]
                ),

            "visible_with_8_chunks":
                int(
                    simulated[8]
                ),

            "selected_ranges":
                json.dumps(
                    selected_ranges,
                    ensure_ascii=False,
                ),

            "changes":
                json.dumps(
                    changes,
                    ensure_ascii=False,
                ),
        })

        if (
            source_index
            % 1000
            == 0
        ):

            print(
                f"\rProcessed "
                f"{source_index:,}/"
                f"{len(groups):,}",
                end="",
                flush=True,
            )

    print()

    # ========================================================
    # SANITY CHECK
    # ========================================================

    if total_noisy == 0:

        raise RuntimeError(
            "No noisy validation documents."
        )

    if total_change_spans == 0:

        raise RuntimeError(
            "No token differences found between "
            "clean/noisy pairs."
        )

    # ========================================================
    # RECALL CONDITIONED ON VISIBILITY
    # ========================================================

    recall_visible = (
        tp_visible
        / any_change_visible
        if any_change_visible
        else None
    )

    recall_not_visible = (
        tp_not_visible
        / no_change_visible
        if no_change_visible
        else None
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "TOKEN-DIFF SANITY CHECK"
    )

    print(
        "=" * 80
    )

    print(
        f"\nNoisy documents                  : "
        f"{total_noisy:,}"
    )

    print(
        f"Identical clean/noisy text       : "
        f"{identical_clean_noisy_text:,}"
    )

    print(
        f"Identical clean/noisy token IDs  : "
        f"{identical_clean_noisy_tokens:,}"
    )

    print(
        f"Total token change spans         : "
        f"{total_change_spans:,}"
    )

    print(
        f"Visible token change spans       : "
        f"{visible_change_spans:,}"
    )

    print(
        f"Invisible token change spans     : "
        f"{invisible_change_spans:,}"
    )

    print(
        "\nDiff opcode distribution:"
    )

    for tag, count in (
        change_opcode_counter
        .most_common()
    ):

        print(
            f"  {tag:10s}: "
            f"{count:,}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "DOCUMENT COVERAGE WITH CURRENT 3 CHUNKS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAt least one changed region visible : "
        f"{any_change_visible:,} "
        f"({percentage(any_change_visible, total_noisy):.2f}%)"
    )

    print(
        f"No changed region visible           : "
        f"{no_change_visible:,} "
        f"({percentage(no_change_visible, total_noisy):.2f}%)"
    )

    if recall_visible is not None:

        print(
            f"\nRecall when change visible     : "
            f"{recall_visible:.6f}"
        )

    if recall_not_visible is not None:

        print(
            f"Recall when change NOT visible : "
            f"{recall_not_visible:.6f}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FALSE NEGATIVE COVERAGE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTotal false negatives      : "
        f"{total_fn:,}"
    )

    print(
        f"FN with visible change     : "
        f"{fn_visible:,} "
        f"({percentage(fn_visible, total_fn):.2f}%)"
    )

    print(
        f"FN with NO visible change  : "
        f"{fn_not_visible:,} "
        f"({percentage(fn_not_visible, total_fn):.2f}%)"
    )

    print(
        "\nFN severity - visible:"
    )

    for severity, count in (
        fn_visible_severity
        .most_common()
    ):

        print(
            f"  {severity:10s}: "
            f"{count:,}"
        )

    print(
        "\nFN severity - NOT visible:"
    )

    for severity, count in (
        fn_not_visible_severity
        .most_common()
    ):

        print(
            f"  {severity:10s}: "
            f"{count:,}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "SINGLE-ERROR DOCUMENTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSingle-error noisy documents: "
        f"{single_error_total:,}"
    )

    print(
        f"Visible                     : "
        f"{single_error_visible:,} "
        f"({percentage(single_error_visible, single_error_total):.2f}%)"
    )

    print(
        f"Not visible                 : "
        f"{single_error_not_visible:,} "
        f"({percentage(single_error_not_visible, single_error_total):.2f}%)"
    )

    print(
        f"\nSingle-error false negatives: "
        f"{single_error_fn:,}"
    )

    print(
        f"FN visible                  : "
        f"{single_error_fn_visible:,} "
        f"({percentage(single_error_fn_visible, single_error_fn):.2f}%)"
    )

    print(
        f"FN not visible              : "
        f"{single_error_fn_not_visible:,} "
        f"({percentage(single_error_fn_not_visible, single_error_fn):.2f}%)"
    )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FALSE NEGATIVE METADATA"
    )

    print(
        "=" * 80
    )

    print(
        "\nFields in FN documents "
        "where change IS visible:"
    )

    for field, count in (
        fn_visible_fields
        .most_common()
    ):

        print(
            f"  {field:10s}: "
            f"{count:,}"
        )

    print(
        "\nFields in FN documents "
        "where NO change is visible:"
    )

    for field, count in (
        fn_not_visible_fields
        .most_common()
    ):

        print(
            f"  {field:10s}: "
            f"{count:,}"
        )

    print(
        "\nTop error types in visible FN docs:"
    )

    for error_type, count in (
        fn_visible_types
        .most_common(
            15
        )
    ):

        print(
            f"  {error_type:25s}: "
            f"{count:,}"
        )

    print(
        "\nTop error types in invisible FN docs:"
    )

    for error_type, count in (
        fn_not_visible_types
        .most_common(
            15
        )
    ):

        print(
            f"  {error_type:25s}: "
            f"{count:,}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "SIMULATED CHUNK COVERAGE"
    )

    print(
        "=" * 80
    )

    print(
        "\nThis is coverage only, NOT predicted model accuracy."
    )

    for candidate in (
        simulated_max_chunks
    ):

        all_covered = (
            simulated_document_coverage[
                candidate
            ]
        )

        fn_covered = (
            simulated_fn_coverage[
                candidate
            ]
        )

        print(
            f"\nMAX_CHUNKS = {candidate}"
        )

        print(
            f"  noisy docs with visible change : "
            f"{all_covered:,} "
            f"({percentage(all_covered, total_noisy):.2f}%)"
        )

        print(
            f"  current FN with visible change : "
            f"{fn_covered:,} "
            f"({percentage(fn_covered, total_fn):.2f}%)"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CHUNK STATISTICS"
    )

    print(
        "=" * 80
    )

    print(
        "\nSelected chunks under current configuration:"
    )

    for count, documents in sorted(
        selected_chunk_distribution
        .items()
    ):

        print(
            f"  {count} chunks : "
            f"{documents:,} documents"
        )

    print(
        "\nTotal possible chunks before limiting:"
    )

    for count, documents in sorted(
        total_chunk_distribution
        .items()
    ):

        print(
            f"  {count:2d} chunks : "
            f"{documents:,} documents"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        DETAIL_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "source_id",
            "variant_id",
            "severity",
            "error_count",
            "probability_error",
            "prediction",
            "is_false_negative",
            "clean_tokens",
            "noisy_tokens",
            "total_chunks",
            "selected_chunks",
            "token_change_spans",
            "any_change_visible",
            "visible_with_3_chunks",
            "visible_with_4_chunks",
            "visible_with_5_chunks",
            "visible_with_6_chunks",
            "visible_with_8_chunks",
            "selected_ranges",
            "changes",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in detail_rows:

            writer.writerow(
                row
            )

    # ========================================================
    # SAVE JSON
    # ========================================================

    result = {
        "dataset":
            "validation",

        "test_used":
            False,

        "method":
            (
                "clean_noisy_token_sequence_diff_"
                "using_model_tokenizer"
            ),

        "locked_threshold":
            threshold,

        "model_config": {
            "text_fields":
                text_fields,

            "field_separator":
                field_separator,

            "max_length":
                max_length,

            "special_tokens":
                special_tokens,

            "content_length":
                content_length,

            "stride":
                stride,

            "step_size":
                step_size,

            "max_chunks":
                max_chunks,
        },

        "sanity": {
            "noisy_documents":
                total_noisy,

            "identical_clean_noisy_text":
                identical_clean_noisy_text,

            "identical_clean_noisy_tokens":
                identical_clean_noisy_tokens,

            "total_change_spans":
                total_change_spans,

            "visible_change_spans":
                visible_change_spans,

            "invisible_change_spans":
                invisible_change_spans,

            "opcode_distribution":
                dict(
                    change_opcode_counter
                ),
        },

        "current_coverage": {
            "visible_documents":
                any_change_visible,

            "invisible_documents":
                no_change_visible,

            "visible_percentage":
                percentage(
                    any_change_visible,
                    total_noisy,
                ),

            "invisible_percentage":
                percentage(
                    no_change_visible,
                    total_noisy,
                ),

            "recall_when_visible":
                recall_visible,

            "recall_when_not_visible":
                recall_not_visible,
        },

        "false_negatives": {
            "total":
                total_fn,

            "visible":
                fn_visible,

            "not_visible":
                fn_not_visible,

            "visible_percentage":
                percentage(
                    fn_visible,
                    total_fn,
                ),

            "not_visible_percentage":
                percentage(
                    fn_not_visible,
                    total_fn,
                ),

            "visible_severity":
                dict(
                    fn_visible_severity
                ),

            "not_visible_severity":
                dict(
                    fn_not_visible_severity
                ),

            "visible_fields":
                dict(
                    fn_visible_fields
                ),

            "not_visible_fields":
                dict(
                    fn_not_visible_fields
                ),

            "visible_error_types":
                dict(
                    fn_visible_types
                ),

            "not_visible_error_types":
                dict(
                    fn_not_visible_types
                ),
        },

        "single_error_documents": {
            "total":
                single_error_total,

            "visible":
                single_error_visible,

            "not_visible":
                single_error_not_visible,

            "false_negative":
                single_error_fn,

            "fn_visible":
                single_error_fn_visible,

            "fn_not_visible":
                single_error_fn_not_visible,
        },

        "simulated_chunk_coverage": {
            str(candidate): {
                "all_noisy_visible":
                    simulated_document_coverage[
                        candidate
                    ],

                "all_noisy_visible_percentage":
                    percentage(
                        simulated_document_coverage[
                            candidate
                        ],
                        total_noisy,
                    ),

                "current_fn_visible":
                    simulated_fn_coverage[
                        candidate
                    ],

                "current_fn_visible_percentage":
                    percentage(
                        simulated_fn_coverage[
                            candidate
                        ],
                        total_fn,
                    ),
            }
            for candidate in simulated_max_chunks
        },

        "selected_chunk_distribution": {
            str(key):
                value
            for key, value in sorted(
                selected_chunk_distribution
                .items()
            )
        },

        "total_chunk_distribution": {
            str(key):
                value
            for key, value in sorted(
                total_chunk_distribution
                .items()
            )
        },
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
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
        "TOKEN-DIFF CHUNK COVERAGE ANALYSIS FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nJSON:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nDetails CSV:\n"
        f"{DETAIL_CSV_PATH}"
    )

    print(
        "\nTest set was NOT read or used."
    )


if __name__ == "__main__":
    main()