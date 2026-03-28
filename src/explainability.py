from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# OpenML credit-g / Statlog German Credit — business-friendly labels (base column names).
FEATURE_BUSINESS: dict[str, tuple[str, str]] = {
    "checking_status": (
        "Checking account balance",
        "How much money the applicant typically holds in a checking account—a common proxy for short-term liquidity.",
    ),
    "duration": (
        "Loan term",
        "How long the borrower has to repay (in months). Longer terms can change monthly burden and total exposure.",
    ),
    "credit_history": (
        "Credit history",
        "Past loans and repayment behavior (e.g., delays or prior defaults).",
    ),
    "purpose": (
        "Loan purpose",
        "What the borrower plans to use the funds for (e.g., car, furniture).",
    ),
    "credit_amount": (
        "Loan amount",
        "Size of the credit line—larger exposures can concentrate loss if the loan goes bad.",
    ),
    "savings_status": (
        "Savings balance",
        "Reported savings—another buffer against missed payments.",
    ),
    "employment": (
        "Employment tenure",
        "How long the applicant has been in their current job or situation.",
    ),
    "installment_commitment": (
        "Installment burden",
        "Existing installment payments as a share of disposable income (higher can mean tighter cash flow).",
    ),
    "personal_status": (
        "Personal / family status",
        "Household structure (e.g., single vs married)—sometimes correlated with stability in historical data.",
    ),
    "other_parties": (
        "Co-applicants or guarantors",
        "Whether someone else is liable for the loan, which can change recovery prospects.",
    ),
    "residence_since": (
        "Years at current address",
        "Stability of housing; frequent moves can correlate with risk in some portfolios.",
    ),
    "property_magnitude": (
        "Property / collateral",
        "Whether the borrower owns assets that could back or motivate repayment.",
    ),
    "age": (
        "Applicant age",
        "Age of the borrower (use with care—must not be used for discriminatory decisions).",
    ),
    "other_payment_plans": (
        "Other repayment plans",
        "Whether other installment plans (e.g., bank plans) are in place.",
    ),
    "housing": (
        "Housing situation",
        "Rent vs own vs free housing—ties to stability and monthly costs.",
    ),
    "existing_credits": (
        "Number of existing credits",
        "How many other loans the applicant is already carrying.",
    ),
    "job": (
        "Job type / skill level",
        "Occupation category (e.g., skilled vs unskilled)—proxy for income stability, not a substitute for income verification.",
    ),
    "num_dependents": (
        "Number of dependents",
        "People financially dependent on the applicant.",
    ),
    "own_telephone": (
        "Telephone listed",
        "Whether a phone is registered (legacy contactability signal in this dataset).",
    ),
    "foreign_worker": (
        "Foreign worker flag",
        "Historical dataset field—must not be used for unfair treatment; included only because the benchmark data contains it.",
    ),
}

# Known categorical column names in credit-g (longest first for parsing one-hot names).
_CAT_BASES_SORTED = sorted(FEATURE_BUSINESS.keys(), key=len, reverse=True)


def _parse_engineered_name(name: str) -> tuple[str, str | None, str]:
    """Return (base_column, onehot_level_or_none, display_technical_name)."""
    if name.startswith("num__"):
        base = name.removeprefix("num__")
        return base, None, name
    if name.startswith("cat__"):
        rest = name.removeprefix("cat__")
        for b in _CAT_BASES_SORTED:
            if rest == b:
                return b, "", name
            prefix = b + "_"
            if rest.startswith(prefix):
                return b, rest[len(prefix) :], name
        return rest, None, name
    return name, None, name


def _business_lines(base: str) -> tuple[str, str]:
    if base in FEATURE_BUSINESS:
        return FEATURE_BUSINESS[base]
    title = base.replace("_", " ").title()
    return title, "Input used by the model after preprocessing."


def _positive_class_sentence(
    target_class_names: tuple[str, ...], clf: Any
) -> tuple[str, str, str]:
    """Human-readable names for each model class label (binary)."""
    classes = getattr(clf, "classes_", None)
    if classes is not None and len(classes) == 2:
        c0, c1 = classes[0], classes[1]
    else:
        c0, c1 = 0, 1

    def _name_for_label(c: Any) -> str:
        try:
            idx = int(c)
            if 0 <= idx < len(target_class_names):
                return target_class_names[idx]
        except (TypeError, ValueError):
            pass
        return str(c)

    name0, name1 = _name_for_label(c0), _name_for_label(c1)
    return (
        name0,
        name1,
        f"The score reflects how the model weighs evidence toward **{name1}** versus **{name0}** "
        f"(probability of **{name1}** rises when the model’s internal score increases).",
    )


def _numeric_direction(
    values: np.ndarray, shap_col: np.ndarray, name0: str, name1: str
) -> str:
    v = np.asarray(values, dtype=float).ravel()
    s = np.asarray(shap_col, dtype=float).ravel()
    mask = np.isfinite(v) & np.isfinite(s)
    if mask.sum() < 10 or np.std(v[mask]) < 1e-12:
        return (
            f"On this sample, the link between higher values and **{name1}** is mixed; "
            "the beeswarm chart shows where cases sit."
        )
    corr = float(np.corrcoef(v[mask], s[mask])[0, 1])
    if corr > 0.08:
        return (
            f"Across the explained applications, **higher values** of this factor tend to push the model "
            f"toward predicting **{name1}** more often."
        )
    if corr < -0.08:
        return (
            f"Across the explained applications, **higher values** of this factor tend to push the model "
            f"toward predicting **{name0}** more often (i.e., lower modeled likelihood of **{name1}**)."
        )
    return (
        f"The relationship is weak in this slice; SHAP still shows **how much** this factor moved each decision, "
        f"not only direction toward **{name1}**."
    )


def _categorical_direction(
    mean_shap: float, mean_abs: float, level: str, name0: str, name1: str
) -> str:
    lvl = level.strip() or "this category"
    if mean_abs > 0.05 and abs(mean_shap) < 0.02:
        return (
            f"This answer (**{lvl}**) **matters a lot** in the model, but it pushes some applicants toward **{name1}** "
            f"and others toward **{name0}** depending on the rest of the profile—the beeswarm chart shows the mix."
        )
    if mean_shap > 0.02:
        return (
            f"For applicants in **{lvl}**, the model tends to shift the score **toward {name1}** "
            f"(relative to other categories of the same field, holding the rest constant)."
        )
    if mean_shap < -0.02:
        return (
            f"For applicants in **{lvl}**, the model tends to shift the score **toward {name0}** "
            f"(relative to other categories—lower modeled **{name1}** likelihood)."
        )
    return (
        f"Category **{lvl}** has a smaller average push in this sample; see the bar chart for overall importance."
    )


def _write_executive_brief(
    path: Path,
    name0: str,
    name1: str,
    score_sentence: str,
    n_samples: int,
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Executive brief: what drove the model on recent applications",
        "",
        "## Business question",
        "",
        "Lenders need to know **which applicant signals** the model leans on when it estimates "
        f"**{name1}** versus **{name0}**. This note translates the model’s top drivers into plain language "
        f"for a **sample of {n_samples} held-out applications** (same cohort used for the SHAP charts).",
        "",
        "## What the model is estimating",
        "",
        score_sentence,
        "",
        "> **Important:** This is a **research / benchmark** pipeline on public data—not a production credit "
        "decision system. Real lending requires policy controls, fair-lending review, and governance. "
        "Some fields in historical data (e.g., demographics) must **not** be used to discriminate.",
        "",
        "## How to read “relation to the outcome”",
        "",
        "- **Influence (importance)** = how much this factor typically moved the model’s score for the people we explained.",
        "- **Direction** = whether, in this sample, the factor is associated with **more** or **less** modeled "
        f"likelihood of **{name1}** (from SHAP sign patterns; not proof of causality).",
        "",
        "## Top drivers (executive view)",
        "",
    ]
    for i, r in enumerate(rows, start=1):
        lines.extend(
            [
                f"### {i}. {r['business_title']}",
                "",
                f"- **What it is:** {r['business_body']}",
                f"- **Technical column:** `{r['technical']}`",
                f"- **Influence:** mean |SHAP| = {r['mean_abs_shap']:.3f} (higher = stronger lever in this model).",
                f"- **Relation to outcome ({name1} vs {name0}):** {r['direction']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Charts",
            "",
            "Use **shap_summary_beeswarm.png** (spread and direction per feature) and **shap_summary_bar.png** "
            "(average impact magnitude) alongside this brief.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def log_shap_explainability(
    best: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    out_dir: Path,
    max_samples: int,
    random_state: int,
    target_class_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """
    SHAP TreeExplainer on the fitted XGBoost step using preprocessed features.
    Writes plots, JSON, and executive_brief.md for MLflow artifacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = best.named_steps["prep"]
    clf = best.named_steps["clf"]

    X_train_t = prep.transform(X_train)
    X_test_t = prep.transform(X_test)
    names = list(prep.get_feature_names_out())

    n = min(max_samples, len(X_test_t))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_test_t), size=n, replace=False)
    X_exp = X_test_t[idx]
    X_exp_df = pd.DataFrame(X_exp, columns=names)

    explainer = shap.TreeExplainer(clf)
    shap_raw = explainer.shap_values(X_exp)
    if isinstance(shap_raw, list):
        shap_values = np.asarray(shap_raw[1])
    else:
        shap_values = np.asarray(shap_raw)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    order = np.argsort(-mean_abs)
    top_idx = order[: min(30, len(names))]

    name0, name1, score_sentence = _positive_class_sentence(target_class_names, clf)

    executive_rows: list[dict[str, Any]] = []
    top: list[dict[str, Any]] = []

    for i in top_idx:
        fname = names[i]
        base, level, technical = _parse_engineered_name(fname)
        biz_title, biz_body = _business_lines(base)
        m_abs = float(mean_abs[i])
        m_sig = float(mean_signed[i])

        if level is None:
            direction = _numeric_direction(X_exp[:, i], shap_values[:, i], name0, name1)
            level_note = ""
        else:
            direction = _categorical_direction(m_sig, m_abs, level or base, name0, name1)
            level_note = f" — level: *{level or '(column)'}*"

        row = {
            "feature": fname,
            "mean_abs_shap": m_abs,
            "mean_shap": m_sig,
            "business_title": biz_title + (level_note if level and level.strip() else ""),
            "business_body": biz_body,
            "technical": technical,
            "direction": direction,
        }
        top.append(
            {
                "feature": fname,
                "mean_abs_shap": m_abs,
                "mean_shap": m_sig,
                "business_title": biz_title,
                "business_what": biz_body,
                "direction_for_executives": direction,
            }
        )
        if len(executive_rows) < 10:
            executive_rows.append(
                {
                    "business_title": row["business_title"],
                    "business_body": biz_body,
                    "technical": technical,
                    "mean_abs_shap": m_abs,
                    "direction": direction,
                }
            )

    summary_path = out_dir / "explainability_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples_explained": int(n),
                "outcome_class_0": name0,
                "outcome_class_1": name1,
                "top_features_by_mean_abs_shap": top,
            },
            f,
            indent=2,
        )

    brief_path = out_dir / "executive_brief.md"
    _write_executive_brief(brief_path, name0, name1, score_sentence, n, executive_rows)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_exp_df,
        show=False,
        max_display=min(20, len(names)),
    )
    plt.tight_layout()
    summary_png = out_dir / "shap_summary_beeswarm.png"
    plt.savefig(summary_png, dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_exp_df,
        plot_type="bar",
        show=False,
        max_display=min(20, len(names)),
    )
    plt.tight_layout()
    bar_png = out_dir / "shap_summary_bar.png"
    plt.savefig(bar_png, dpi=120, bbox_inches="tight")
    plt.close()

    return {
        "n_samples": int(n),
        "summary_json": str(summary_path),
        "executive_brief_md": str(brief_path),
        "beeswarm_png": str(summary_png),
        "bar_png": str(bar_png),
    }
