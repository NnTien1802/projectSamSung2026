from pathlib import Path
import json
import re


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_ROOT = (
    MODULE_ROOT
    / "scripts"
    / "consistency"
)

CACHE_REPORT_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "mil_chunk_cache_mbert_v1"
    / "cache_report.json"
)

TEST_CACHE_REPORT_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "mil_chunk_cache_mbert_v1"
    / "test_cache_report.json"
)


# ============================================================
# SEARCH TERMS
# ============================================================

SEARCH_TERMS = [
    "linspace",
    "round",
    "rint",
    "ceil",
    "floor",
    "chunk",
    "max_chunks",
    "chunk_tokens",
    "overlap",
    "stride",
    "start",
    "sample",
    "uniform",
]


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def print_json_section(
    title,
    data,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
    )


def relative(path):

    try:

        return str(
            path.relative_to(
                MODULE_ROOT
            )
        )

    except ValueError:

        return str(path)


# ============================================================
# CACHE REPORT
# ============================================================

def inspect_cache_reports():

    print()
    print("=" * 80)
    print(
        "A. CACHE REPORT"
    )
    print("=" * 80)

    if CACHE_REPORT_PATH.exists():

        report = load_json(
            CACHE_REPORT_PATH
        )

        print()
        print(
            f"File: "
            f"{relative(CACHE_REPORT_PATH)}"
        )

        if (
            "chunk_policy"
            in report
        ):

            print_json_section(
                "chunk_policy",
                report[
                    "chunk_policy"
                ],
            )

        else:

            print()
            print(
                "No top-level chunk_policy key."
            )

            print()
            print(
                "Full report:"
            )

            print(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                )
            )

    else:

        print(
            f"Missing: "
            f"{CACHE_REPORT_PATH}"
        )

    print()
    print("=" * 80)
    print(
        "B. TEST CACHE REPORT"
    )
    print("=" * 80)

    if TEST_CACHE_REPORT_PATH.exists():

        report = load_json(
            TEST_CACHE_REPORT_PATH
        )

        print()
        print(
            f"File: "
            f"{relative(TEST_CACHE_REPORT_PATH)}"
        )

        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
        )

    else:

        print(
            f"Missing: "
            f"{TEST_CACHE_REPORT_PATH}"
        )


# ============================================================
# SOURCE CODE SEARCH
# ============================================================

def inspect_source_code():

    print()
    print("=" * 80)
    print(
        "C. SEARCH CONSISTENCY SOURCE CODE"
    )
    print("=" * 80)

    if not SCRIPTS_ROOT.exists():

        print(
            f"Missing scripts root: "
            f"{SCRIPTS_ROOT}"
        )

        return

    python_files = sorted(
        SCRIPTS_ROOT.rglob(
            "*.py"
        )
    )

    print()
    print(
        f"Python files found: "
        f"{len(python_files):,}"
    )

    relevant_files = []

    for path in python_files:

        try:

            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except Exception:

            continue

        lowered = text.lower()

        score = 0

        for term in SEARCH_TERMS:

            if term in lowered:

                score += 1

        # Strong preference for chunk/cache/MIL scripts
        path_lower = str(
            path
        ).lower()

        if "chunk" in path_lower:

            score += 5

        if "cache" in path_lower:

            score += 5

        if "mil" in path_lower:

            score += 5

        if score > 0:

            relevant_files.append(
                (
                    score,
                    path,
                    text,
                )
            )

    relevant_files.sort(
        key=lambda item: (
            -item[0],
            str(item[1]),
        )
    )

    print()
    print(
        "Relevant files:"
    )

    for score, path, _ in relevant_files:

        print(
            f"  score={score:02d}  "
            f"{relative(path)}"
        )

    print()
    print("=" * 80)
    print(
        "D. RELEVANT SOURCE LINES"
    )
    print("=" * 80)

    patterns = [
        r"linspace",
        r"\.round",
        r"\brint\b",
        r"\bceil\b",
        r"\bfloor\b",
        r"chunk[_ ]?tokens",
        r"chunk[_ ]?size",
        r"max[_ ]?chunks",
        r"chunk[_ ]?overlap",
        r"stride",
        r"start",
        r"uniform",
    ]

    regex = re.compile(
        "|".join(
            patterns
        ),
        flags=re.IGNORECASE,
    )

    printed_any = False

    for score, path, text in relevant_files:

        lines = text.splitlines()

        matching_lines = []

        for index, line in enumerate(
            lines,
            start=1,
        ):

            if regex.search(
                line
            ):

                matching_lines.append(
                    index
                )

        if not matching_lines:

            continue

        printed_any = True

        # Expand around matching lines
        selected_lines = set()

        for line_number in matching_lines:

            for expanded in range(
                max(
                    1,
                    line_number - 4,
                ),
                min(
                    len(lines),
                    line_number + 4,
                )
                + 1,
            ):

                selected_lines.add(
                    expanded
                )

        print()
        print("-" * 80)
        print(
            relative(path)
        )
        print("-" * 80)

        previous = None

        for line_number in sorted(
            selected_lines
        ):

            if (
                previous is not None
                and
                line_number
                >
                previous + 1
            ):

                print(
                    "    ..."
                )

            print(
                f"{line_number:5d}: "
                f"{lines[line_number - 1]}"
            )

            previous = line_number

    if not printed_any:

        print()
        print(
            "No relevant source lines found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "INTEGRATION STEP 06"
    )
    print(
        "INSPECT ORIGINAL MAX-MIL CHUNK POLICY"
    )
    print("=" * 80)

    print()
    print(
        "READ-ONLY."
    )

    print(
        "NO TRAINING."
    )

    print(
        "NO INFERENCE."
    )

    print(
        "NO TEST TUNING."
    )

    inspect_cache_reports()

    inspect_source_code()

    print()
    print("=" * 80)
    print(
        "WHAT WE NEED TO IDENTIFY"
    )
    print("=" * 80)

    print()
    print(
        "Look for the exact code used to reduce "
        "more than 6 raw windows to 6 selected chunks."
    )

    print()
    print(
        "Examples:"
    )

    print(
        "  np.linspace(...).round().astype(int)"
    )

    print(
        "  np.rint(np.linspace(...)).astype(int)"
    )

    print(
        "  np.ceil(np.linspace(...)).astype(int)"
    )

    print(
        "  another explicit index formula"
    )

    print()
    print(
        "INTEGRATION STEP 06 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()