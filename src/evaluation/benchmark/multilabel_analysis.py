"""
Multi-label failure analysis for TravelPlanEdit benchmark.

Replaces the mutually-exclusive failure reason_code with non-mutually-exclusive
per-sample flags: format_fail, feasibility_fail, hard_pres_fail, soft_pres_fail,
edit_fail. Also classifies feasibility violations into fine-grained fact-grounding
categories (nonexistent_poi, wrong_fact, schedule_issue, transport_issue, etc.).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Violation code → semantic category mapping
# ---------------------------------------------------------------------------

VIOLATION_CATEGORY_MAP: Dict[str, str] = {
    # True fact grounding — POI does not exist in the database
    "invalid_poi": "nonexistent_poi",
    # True fact grounding — wrong metadata about an existing POI
    "opening_hours_violation": "wrong_fact",
    "city_mismatch": "wrong_fact",
    # Schedule / temporal feasibility
    "timeline_overlap": "schedule_issue",
    "early_attraction": "schedule_issue",
    "dense_day_attractions": "schedule_issue",
    "large_idle_gap": "schedule_issue",
    "day_boundary_transfer": "schedule_issue",
    # Transport feasibility
    "long_walk_distance": "transport_issue",
    "long_walk_duration": "transport_issue",
    "transport_before_previous_end": "transport_issue",
    "transport_after_activity_start": "transport_issue",
    "post_transport_idle_gap": "transport_issue",
    # Missing required elements
    "missing_accommodation": "missing_required",
    "missing_people_number": "missing_required",
    "missing_itinerary": "missing_required",
    # Data quality
    "duplicate_poi": "duplicate",
    "invalid_time_value": "format_issue",
}

FACT_GROUNDING_CATEGORIES = {"nonexistent_poi", "wrong_fact"}


def classify_violation_code(code: str) -> str:
    """Map a violation code string to its semantic category."""
    return VIOLATION_CATEGORY_MAP.get(code, "other")


def collect_violation_codes(result: Dict[str, Any]) -> List[str]:
    """Extract all violation codes from a benchmark result's feasibility section."""
    codes: List[str] = []
    level1 = result.get("level1", {})
    if not isinstance(level1, dict):
        return codes
    feasibility = level1.get("feasibility", {})
    if not isinstance(feasibility, dict):
        return codes
    for section_key in ("hygiene_violations", "quality_violations"):
        for item in feasibility.get(section_key, []) or []:
            if isinstance(item, dict):
                code = item.get("code")
                if isinstance(code, str) and code:
                    codes.append(code)
    return codes


def classify_feasibility_categories(result: Dict[str, Any]) -> Dict[str, int]:
    """Count violation codes by semantic category for a single sample."""
    counts: Dict[str, int] = defaultdict(int)
    for code in collect_violation_codes(result):
        category = classify_violation_code(code)
        counts[category] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Multi-label flags (non-mutually-exclusive)
# ---------------------------------------------------------------------------


def compute_multilabel_flags(result: Dict[str, Any]) -> Dict[str, bool]:
    """Compute non-mutually-exclusive failure flags for a single sample.

    Each flag is independent — a sample can have any combination.
    """
    run_context = result.get("run_context", {})
    if not isinstance(run_context, dict):
        run_context = {}

    run_success = bool(run_context.get("success", True))
    run_errors = run_context.get("errors", [])
    if not isinstance(run_errors, list):
        run_errors = []

    format_fail = (not run_success) or bool(run_errors)

    level1 = result.get("level1", {})
    if not isinstance(level1, dict):
        level1 = {}

    feasibility = level1.get("feasibility", {}) if isinstance(level1.get("feasibility"), dict) else {}
    hard_pres = level1.get("origin_logical_preservation", {}) if isinstance(level1.get("origin_logical_preservation"), dict) else {}
    soft_pres = level1.get("origin_preference_preservation", {}) if isinstance(level1.get("origin_preference_preservation"), dict) else {}

    feasibility_fail = not bool(feasibility.get("pass", True))
    hard_pres_fail = not bool(hard_pres.get("pass", True))
    soft_pres_fail = not bool(soft_pres.get("pass", True))

    level2 = result.get("level2", {})
    if not isinstance(level2, dict):
        level2 = {}
    edit_fail = bool(level2) and not bool(level2.get("pass", True))
    # Also flag if L2 logical or preference targets exist and fail
    edit_logical_success = level2.get("edit_logical_success", {}) if isinstance(level2.get("edit_logical_success"), dict) else {}
    edit_pref_success = level2.get("edit_preference_success", {}) if isinstance(level2.get("edit_preference_success"), dict) else {}
    edit_logical_fail = edit_logical_success.get("supported_constraints", 0) > 0 and not bool(edit_logical_success.get("pass", True))
    edit_pref_fail = edit_pref_success.get("supported_preferences", 0) > 0 and not bool(edit_pref_success.get("pass", True))

    return {
        "format_fail": format_fail,
        "feasibility_fail": feasibility_fail,
        "hard_pres_fail": hard_pres_fail,
        "soft_pres_fail": soft_pres_fail,
        "edit_fail": edit_fail,
        "edit_logical_fail": edit_logical_fail,
        "edit_pref_fail": edit_pref_fail,
        "all_pass": not any([format_fail, feasibility_fail, hard_pres_fail, soft_pres_fail, edit_fail]),
    }


def multilabel_flag_names() -> List[str]:
    """Return ordered list of primary multi-label flag names."""
    return ["format_fail", "feasibility_fail", "hard_pres_fail", "soft_pres_fail", "edit_fail"]


# ---------------------------------------------------------------------------
# Fact-grounding vs origin-preservation overlap analysis
# ---------------------------------------------------------------------------


def analyze_fact_grounding_overlap(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check how many fact-grounding failures also violate origin preservation.

    Answers: is preservation being "absorbed" by feasibility checks?
    If a high proportion of samples with fact-grounding feasibility failures
    also fail origin preservation, it means feasibility is catching what is
    fundamentally a preservation problem.
    """
    total = len(results)
    fact_grounding_samples = 0
    fact_grounding_codes: Dict[str, int] = defaultdict(int)

    # Among fact-grounding failures, how many also fail preservation?
    fg_with_hard_pres_fail = 0
    fg_with_soft_pres_fail = 0
    fg_with_any_pres_fail = 0
    fg_only = 0  # fact grounding fail but preservation passes

    # Detailed breakdown by violation sub-category
    subcategory_overlap: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "hard_pres_fail": 0, "soft_pres_fail": 0, "only_feas": 0}
    )

    nonexistent_poi_count = 0
    wrong_fact_count = 0

    for result in results:
        violation_categories = classify_feasibility_categories(result)
        has_fact_grounding = bool(
            violation_categories.get("nonexistent_poi", 0)
            + violation_categories.get("wrong_fact", 0)
        )

        if not has_fact_grounding:
            continue

        fact_grounding_samples += 1
        for cat in FACT_GROUNDING_CATEGORIES:
            count = violation_categories.get(cat, 0)
            if count > 0:
                fact_grounding_codes[cat] += count

        flags = compute_multilabel_flags(result)

        if flags["hard_pres_fail"]:
            fg_with_hard_pres_fail += 1
        if flags["soft_pres_fail"]:
            fg_with_soft_pres_fail += 1
        if flags["hard_pres_fail"] or flags["soft_pres_fail"]:
            fg_with_any_pres_fail += 1
        if not flags["hard_pres_fail"] and not flags["soft_pres_fail"]:
            fg_only += 1

        for cat in FACT_GROUNDING_CATEGORIES:
            if violation_categories.get(cat, 0) > 0:
                subcategory_overlap[cat]["total"] += 1
                if flags["hard_pres_fail"]:
                    subcategory_overlap[cat]["hard_pres_fail"] += 1
                if flags["soft_pres_fail"]:
                    subcategory_overlap[cat]["soft_pres_fail"] += 1
                if not flags["hard_pres_fail"] and not flags["soft_pres_fail"]:
                    subcategory_overlap[cat]["only_feas"] += 1

        nonexistent_poi_count += violation_categories.get("nonexistent_poi", 0)
        wrong_fact_count += violation_categories.get("wrong_fact", 0)

    total_violations = nonexistent_poi_count + wrong_fact_count

    return {
        "total_samples": total,
        "fact_grounding_samples": fact_grounding_samples,
        "fg_sample_rate": fact_grounding_samples / total if total else 0.0,
        "nonexistent_poi_violations": nonexistent_poi_count,
        "wrong_fact_violations": wrong_fact_count,
        "nonexistent_poi_ratio": nonexistent_poi_count / total_violations if total_violations else 0.0,
        "wrong_fact_ratio": wrong_fact_count / total_violations if total_violations else 0.0,
        "fg_with_hard_pres_fail": fg_with_hard_pres_fail,
        "fg_with_soft_pres_fail": fg_with_soft_pres_fail,
        "fg_with_any_pres_fail": fg_with_any_pres_fail,
        "fg_only_feasibility": fg_only,
        "preservation_overlap_rate": (
            fg_with_any_pres_fail / fact_grounding_samples if fact_grounding_samples else 0.0
        ),
        "subcategory_overlap": {
            cat: dict(data) for cat, data in subcategory_overlap.items()
        },
        "verdict": _fact_grounding_verdict(
            nonexistent_poi_count, wrong_fact_count,
            fg_with_any_pres_fail, fact_grounding_samples,
        ),
    }


def _fact_grounding_verdict(
    nonexistent: int,
    wrong_fact: int,
    overlap: int,
    total_fg: int,
) -> str:
    """Generate a verdict string for the fact grounding diagnostic."""
    parts: List[str] = []
    total = nonexistent + wrong_fact
    if total == 0:
        return "no fact-grounding violations found"

    nonexistent_pct = nonexistent / total * 100
    wrong_pct = wrong_fact / total * 100
    overlap_pct = overlap / total_fg * 100 if total_fg else 0.0

    parts.append(
        f"violation composition: nonexistent_poi={nonexistent_pct:.1f}%, "
        f"wrong_fact={wrong_pct:.1f}%"
    )
    parts.append(
        f"preservation overlap: {overlap_pct:.1f}% of fact-grounding samples "
        f"also fail origin preservation"
    )

    if nonexistent_pct < 30:
        parts.append("WARNING: 'nonexistent_poi' rate is low — 'fact grounding' naming may be misleading")
    if overlap_pct > 50:
        parts.append("WARNING: high preservation overlap — feasibility absorbs preservation signal")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Multi-label frequency analysis
# ---------------------------------------------------------------------------


def compute_multilabel_frequencies(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build multi-label frequency distribution across all samples.

    Returns:
        label_freq: per-label frequency
        cooccurrence: pairwise co-occurrence matrix
        combination_freq: frequency of each label combination pattern
        label_freq_by_scope: per-label frequency broken down by edit scope
        sample_details: per-sample multi-label flags
    """
    flag_names = multilabel_flag_names()
    label_counts: Dict[str, int] = defaultdict(int)
    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    combination_counts: Counter = Counter()
    samples: List[Dict[str, Any]] = []
    scope_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for result in results:
        flags = compute_multilabel_flags(result)
        active = [name for name in flag_names if flags.get(name)]
        label_key = "+".join(active) if active else "all_pass"

        for name in flag_names:
            if flags.get(name):
                label_counts[name] += 1

        for i, name_i in enumerate(flag_names):
            for name_j in flag_names[i + 1 :]:
                if flags.get(name_i) and flags.get(name_j):
                    cooccurrence[(name_i, name_j)] += 1

        combination_counts[label_key] += 1

        # Scope breakdown
        scope = "unknown"
        conflict_labels = result.get("conflict_labels")
        if isinstance(conflict_labels, list) and conflict_labels:
            first = conflict_labels[0]
            if isinstance(first, list) and first:
                scope = str(first[0])
        elif isinstance(result.get("_scope"), str):
            scope = result["_scope"]

        for name in flag_names:
            if flags.get(name):
                scope_counts[scope][name] += 1

        samples.append({
            "record_id": result.get("record_id", "unknown"),
            "flags": flags,
            "active_labels": active if active else ["all_pass"],
            "scope": scope,
        })

    total = len(results)
    label_freq = {
        name: {
            "count": label_counts.get(name, 0),
            "rate": label_counts.get(name, 0) / total if total else 0.0,
        }
        for name in flag_names
    }

    cooccurrence_matrix = {}
    for (a, b), count in cooccurrence.items():
        denom = max(label_counts.get(a, 1), label_counts.get(b, 1))
        cooccurrence_matrix[f"{a} & {b}"] = {
            "count": count,
            "jaccard": count / denom if denom else 0.0,
        }

    scope_freq = {}
    for scope, counts in scope_counts.items():
        scope_total = sum(
            1 for s in samples if s["scope"] == scope
        )
        scope_freq[scope] = {
            name: {
                "count": counts.get(name, 0),
                "rate": counts.get(name, 0) / scope_total if scope_total else 0.0,
            }
            for name in flag_names
        }

    top_combinations = combination_counts.most_common(20)

    return {
        "total_samples": total,
        "label_frequency": label_freq,
        "cooccurrence": cooccurrence_matrix,
        "combination_frequency": [
            {"labels": combo, "count": count, "rate": count / total if total else 0.0}
            for combo, count in top_combinations
        ],
        "label_frequency_by_scope": scope_freq,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Full diagnostic report (combines all analyses)
# ---------------------------------------------------------------------------


def run_full_multilabel_analysis(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all multi-label analyses and return a combined diagnostic report."""
    fg_overlap = analyze_fact_grounding_overlap(results)
    multilabel_freq = compute_multilabel_frequencies(results)

    # Overall breakdown of violation categories
    all_category_counts: Dict[str, int] = defaultdict(int)
    for result in results:
        for cat, count in classify_feasibility_categories(result).items():
            all_category_counts[cat] += count

    # Count samples with feasibility failures that have NO fact-grounding codes
    feasibility_fail_no_fg = 0
    feasibility_fail_with_fg = 0
    for result in results:
        flags = compute_multilabel_flags(result)
        if not flags["feasibility_fail"]:
            continue
        violation_cats = classify_feasibility_categories(result)
        has_fg = bool(
            violation_cats.get("nonexistent_poi", 0)
            + violation_cats.get("wrong_fact", 0)
        )
        if has_fg:
            feasibility_fail_with_fg += 1
        else:
            feasibility_fail_no_fg += 1

    return {
        "fact_grounding_diagnostic": fg_overlap,
        "multilabel_frequency": multilabel_freq,
        "violation_category_distribution": dict(all_category_counts),
        "feasibility_fail_breakdown": {
            "total_feasibility_failures": feasibility_fail_no_fg + feasibility_fail_with_fg,
            "with_fact_grounding": feasibility_fail_with_fg,
            "without_fact_grounding": feasibility_fail_no_fg,
            "fact_grounding_fraction": (
                feasibility_fail_with_fg
                / (feasibility_fail_no_fg + feasibility_fail_with_fg)
                if (feasibility_fail_no_fg + feasibility_fail_with_fg)
                else 0.0
            ),
        },
    }


# ---------------------------------------------------------------------------
# Jaccard matrix helper
# ---------------------------------------------------------------------------


def build_jaccard_matrix(freq: Dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Build a 5x5 Jaccard co-occurrence matrix from multilabel frequency data."""
    flag_names = multilabel_flag_names()
    n = len(flag_names)
    label_freq = freq["label_frequency"]
    cooccurrence = freq["cooccurrence"]

    matrix = np.zeros((n, n))
    for i, name_i in enumerate(flag_names):
        for j, name_j in enumerate(flag_names):
            if i == j:
                matrix[i, j] = 1.0
                continue
            key = (
                f"{name_i} & {name_j}"
                if f"{name_i} & {name_j}" in cooccurrence
                else f"{name_j} & {name_i}"
            )
            pair = cooccurrence.get(key, {})
            matrix[i, j] = pair.get("jaccard", 0.0)

    return flag_names, matrix


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ensure_output_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)


DISPLAY_NAMES = {
    "format_fail": "Format",
    "feasibility_fail": "Feasibility",
    "hard_pres_fail": "Hard Pres.",
    "soft_pres_fail": "Soft Pres.",
    "edit_fail": "Edit",
}

# Color per bar, drawn from the style guide main palette
BAR_DISPLAY_COLORS = {
    "Format":      "#9A9A9A",  # Neutral Gray
    "Feasibility": "#D98C3A",  # Muted Amber
    "Hard Pres.":  "#3B6EA8",  # Academic Blue
    "Soft Pres.":  "#C65A5A",  # Soft Red
    "Edit":        "#6FA36F",  # Sage Green
}

# Style-guide unified teal sequential colormap for heatmaps
HEATMAP_TEAL_COLORS = ["#F8FAFA", "#DDEDEA", "#AFCFCA", "#6FA8A3", "#2F6F73"]


def _luminance(hex_color: str) -> float:
    """Return perceptual luminance of a hex color (0–1)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _setup_paper_rc(double_col: bool = True) -> None:
    """Override rcParams for paper figures: Times New Roman, ACL sizes."""
    import matplotlib.pyplot as plt
    from experiments.main_analysis.plotting_rc import apply_morandi_style
    apply_morandi_style()
    if double_col:
        plt.rcParams.update({
            "font.size": 10.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
        })
    else:
        plt.rcParams.update({
            "font.size": 9,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
        })


# ---------------------------------------------------------------------------
# Figure A: Multi-label failure frequency
# ---------------------------------------------------------------------------


def plot_figure_a_frequency(
    multilabel_freq: Dict[str, Any],
    output_dir: str,
    *,
    prefix: str = "fig_a",
) -> str:
    """Figure A: Multi-label failure frequency bar chart (single-column)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from experiments.main_analysis.plotting_rc import save_figure, style_axes

    _setup_paper_rc(double_col=False)

    flag_names = multilabel_flag_names()
    label_freq = multilabel_freq["label_frequency"]
    total = multilabel_freq["total_samples"]

    names = [DISPLAY_NAMES[f] for f in flag_names]
    rates = [label_freq[f]["rate"] * 100 for f in flag_names]
    counts = [label_freq[f]["count"] for f in flag_names]
    colors = [BAR_DISPLAY_COLORS[n] for n in names]

    all_pass_count = sum(
        1 for s in multilabel_freq.get("samples", [])
        if s.get("active_labels") == ["all_pass"]
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.8), constrained_layout=True)

    x = np.arange(len(names))
    bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.8, width=0.58)

    for i, (rate, count) in enumerate(zip(rates, counts)):
        ax.text(i, rate + 1.2, f"{count}\n{rate:.1f}%", ha="center",
                fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Failure rate (%)")
    ax.set_ylim(0, max(rates) * 1.22 if rates else 100)

    style_axes(ax)
    path = save_figure(fig, f"{prefix}_frequency", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure B: Failure co-occurrence heatmap (Jaccard)
# ---------------------------------------------------------------------------


def plot_figure_b_cooccurrence_heatmap(
    multilabel_freq: Dict[str, Any],
    output_dir: str,
    *,
    prefix: str = "fig_b",
) -> str:
    """Figure B: Failure co-occurrence heatmap using Jaccard similarity.

    Uses the style-guide unified teal sequential colormap.
    Diagonal cells show individual failure rates; off-diagonal cells show Jaccard.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from experiments.main_analysis.plotting_rc import save_figure, style_axes

    _setup_paper_rc(double_col=True)

    flag_names = multilabel_flag_names()
    short_names = ["Format", "Feasibility", "Hard Pres.", "Soft Pres.", "Edit"]
    label_freq = multilabel_freq["label_frequency"]
    cooccurrence = multilabel_freq["cooccurrence"]

    n = len(flag_names)
    jaccard_mat = np.zeros((n, n))
    diag_rates = np.zeros(n)

    for i, name_i in enumerate(flag_names):
        diag_rates[i] = label_freq[name_i]["rate"] * 100
        for j, name_j in enumerate(flag_names):
            if i == j:
                jaccard_mat[i, j] = 1.0
                continue
            key = (
                f"{name_i} & {name_j}"
                if f"{name_i} & {name_j}" in cooccurrence
                else f"{name_j} & {name_i}"
            )
            pair = cooccurrence.get(key, {})
            jaccard_mat[i, j] = pair.get("jaccard", 0.0)

    teal_cmap = LinearSegmentedColormap.from_list("tpe_teal", HEATMAP_TEAL_COLORS, N=256)

    fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)

    # Show full matrix (diagonal included) with the teal colormap.
    # Diagonal will display as the deepest teal (value=1.0).
    im = ax.imshow(jaccard_mat, cmap=teal_cmap, vmin=0.0, vmax=1.0, aspect="equal")

    # Annotate every cell
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: show failure rate
                fr = diag_rates[i]
                ax.text(j, i, f"{fr:.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
            else:
                jv = jaccard_mat[i, j]
                cell_bg = teal_cmap(jv)
                lum = _luminance(cell_bg) if hasattr(cell_bg, "__len__") else 0.5
                # matplotlib colormap returns RGBA tuple
                r, g, b, _ = cell_bg
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if lum < 0.55 else "#333333"
                ax.text(j, i, f"{jv:.3f}", ha="center", va="center",
                        fontsize=10,
                        fontweight="bold" if jv >= 0.6 else "normal",
                        color=text_color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=18, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.80)
    cbar.set_label("Jaccard", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    style_axes(ax)

    path = save_figure(fig, f"{prefix}_cooccurrence_heatmap", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure C: Fact-grounding diagnostic (composition + overlap)
# ---------------------------------------------------------------------------


def plot_figure_c_fact_grounding(
    fg_diagnostic: Dict[str, Any],
    output_dir: str,
    *,
    prefix: str = "fig_c",
) -> str:
    """Figure C: Fact-grounding violation composition and preservation overlap."""
    import matplotlib.pyplot as plt
    from experiments.main_analysis.plotting_rc import save_figure, style_axes

    _setup_paper_rc(double_col=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    nonexistent = fg_diagnostic["nonexistent_poi_violations"]
    wrong_fact = fg_diagnostic["wrong_fact_violations"]
    total_v = nonexistent + wrong_fact

    if total_v > 0:
        pie_vals = [nonexistent, wrong_fact]
        pie_labels = [f"Nonexistent POI ({nonexistent})", f"Wrong Fact ({wrong_fact})"]
        pie_colors = ["#C65A5A", "#D98C3A"]  # Soft Red, Muted Amber
        wedges, texts, autotexts = ax1.pie(
            pie_vals, labels=None, colors=pie_colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.72,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax1.legend(wedges, pie_labels, loc="center left",
                   bbox_to_anchor=(0.95, 0.5), fontsize=9,
                   frameon=True, fancybox=False, edgecolor="#CCCCCC")
    else:
        ax1.text(0.5, 0.5, "No fact-grounding\nviolations found",
                transform=ax1.transAxes, ha="center", va="center", fontsize=9)
    ax1.text(-0.15, 1.08, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")

    # --- Right: overlap ---
    fg_total = fg_diagnostic["fact_grounding_samples"]
    only_feas = fg_diagnostic["fg_only_feasibility"]
    with_pres = fg_diagnostic["fg_with_any_pres_fail"]

    if fg_total > 0:
        categories = ["FG only\n(isolated)", "FG + Pres.\n(overlap)"]
        values = [only_feas, with_pres]
        colors2 = ["#6FA36F", "#C65A5A"]  # Sage Green, Soft Red
        x_pos = [0.2, 0.8]
        bars = ax2.bar(x_pos, values, color=colors2, edgecolor="white",
                       linewidth=0.8, width=0.32)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, fontsize=9.5)
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylabel("Number of samples")
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val}\n({val/fg_total*100:.1f}%)", ha="center", fontsize=9)
        ax2.set_ylim(0, max(values) * 1.28)
    else:
        ax2.text(0.5, 0.5, "No fact-grounding\nsamples to analyze",
                transform=ax2.transAxes, ha="center", va="center", fontsize=9)
    ax2.text(-0.15, 1.08, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    style_axes(ax1)
    style_axes(ax2)
    path = save_figure(fig, f"{prefix}_fact_grounding", output_dir)
    plt.close(fig)
    return path
