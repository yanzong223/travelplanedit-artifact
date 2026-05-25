"""Run hygiene and quality filtering over origin plan folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_clean.rules import (  # type: ignore[no-redef]
        HYGIENE_CONFIG,
        QUALITY_GATE_CONFIG,
        evaluate_plan,
    )
else:
    from data_clean.rules import (
        HYGIENE_CONFIG,
        QUALITY_GATE_CONFIG,
        evaluate_plan,
    )


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Audit origin plans with hygiene and quality gate rules."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=workspace_root / "data" / "origin_plan",
        help="Root directory that contains origin plan source folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "origin_plan_filter_v1",
        help="Directory to save summary tables and detailed manifests.",
    )
    return parser.parse_args()


def source_directories(input_root: Path) -> List[Path]:
    if any(input_root.glob("*.json")):
        return [input_root]

    directories: List[Path] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.json")):
            directories.append(child)
    return directories


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    global_hygiene_reasons: Counter[str] = Counter()
    global_quality_reasons: Counter[str] = Counter()

    for source_dir in source_directories(input_root):
        hygiene_reason_counts: Counter[str] = Counter()
        quality_reason_counts: Counter[str] = Counter()
        hygiene_pass_files: List[str] = []
        hygiene_fail_files: List[str] = []
        quality_pass_files: List[str] = []
        quality_fail_files: List[str] = []

        for file_path in sorted(source_dir.glob("*.json")):
            plan = load_json(file_path)
            evaluation = evaluate_plan(plan)

            hygiene_codes = sorted({item.code for item in evaluation.hygiene_violations})
            quality_codes = sorted({item.code for item in evaluation.quality_violations})

            if evaluation.hygiene_pass:
                hygiene_pass_files.append(file_path.name)
            else:
                hygiene_fail_files.append(file_path.name)
                hygiene_reason_counts.update(hygiene_codes)
                global_hygiene_reasons.update(hygiene_codes)

            if evaluation.quality_pass:
                quality_pass_files.append(file_path.name)
            else:
                quality_fail_files.append(file_path.name)
                if evaluation.hygiene_pass:
                    quality_reason_counts.update(quality_codes)
                    global_quality_reasons.update(quality_codes)

            detail_rows.append(
                {
                    "source": source_dir.name,
                    "file": file_path.name,
                    "hygiene_pass": evaluation.hygiene_pass,
                    "quality_pass": evaluation.quality_pass,
                    "hygiene_violation_codes": hygiene_codes,
                    "quality_violation_codes": quality_codes,
                    "hygiene_violations": [
                        item.to_dict() for item in evaluation.hygiene_violations
                    ],
                    "quality_violations": [
                        item.to_dict() for item in evaluation.quality_violations
                    ],
                }
            )

        total = len(hygiene_pass_files) + len(hygiene_fail_files)
        summary_rows.append(
            {
                "source": source_dir.name,
                "total_files": total,
                "hygiene_pass": len(hygiene_pass_files),
                "hygiene_fail": len(hygiene_fail_files),
                "quality_pass": len(quality_pass_files),
                "quality_fail": len(quality_fail_files),
                "quality_fail_after_hygiene": len(hygiene_pass_files)
                - len(quality_pass_files),
                "hygiene_pass_rate": round(len(hygiene_pass_files) / total, 4)
                if total
                else 0,
                "quality_pass_rate": round(len(quality_pass_files) / total, 4)
                if total
                else 0,
                "top_hygiene_reasons": dict(hygiene_reason_counts.most_common()),
                "top_quality_reasons": dict(quality_reason_counts.most_common()),
            }
        )

        source_output_dir = output_dir / "manifests" / source_dir.name
        write_json(
            source_output_dir / "summary.json",
            {
                "source": source_dir.name,
                "total_files": total,
                "hygiene_pass": hygiene_pass_files,
                "hygiene_fail": hygiene_fail_files,
                "quality_pass": quality_pass_files,
                "quality_fail": quality_fail_files,
                "hygiene_reason_counts": dict(hygiene_reason_counts),
                "quality_reason_counts": dict(quality_reason_counts),
            },
        )

    summary_rows.sort(key=lambda row: row["source"])
    write_json(
        output_dir / "summary.json",
        {
            "input_root": str(input_root),
            "rule_configs": {
                "hygiene": HYGIENE_CONFIG.to_dict(),
                "quality_gate": QUALITY_GATE_CONFIG.to_dict(),
            },
            "summary": summary_rows,
            "global_hygiene_fail_reasons": dict(global_hygiene_reasons),
            "global_quality_fail_reasons": dict(global_quality_reasons),
        },
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "source",
            "total_files",
            "hygiene_pass",
            "hygiene_fail",
            "quality_pass",
            "quality_fail",
            "quality_fail_after_hygiene",
            "hygiene_pass_rate",
            "quality_pass_rate",
            "top_hygiene_reasons",
            "top_quality_reasons",
        ],
    )
    write_jsonl(output_dir / "details.jsonl", detail_rows)

    compact_rows = [
        {
            "source": row["source"],
            "total_files": row["total_files"],
            "hygiene_pass": row["hygiene_pass"],
            "quality_pass": row["quality_pass"],
        }
        for row in summary_rows
    ]
    print(json.dumps(compact_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
