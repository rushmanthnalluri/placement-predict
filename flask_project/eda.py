"""EDA artifact builders for the Placement Predict pipeline.

Loads the active dataset (uploaded file or the bundled 50k workbook) once,
caches the DataFrame, and computes every number the EDA stage pages need —
all as plain Python types so templates can serialize them with `tojson`.

Mirrors eda.ipynb: missing-value analysis, describe, outlier box stats,
correlation matrix, mean imputation, z-score standardization, correlations
with PlacementStatus, and per-category placement rates.
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column schema (mirrors the notebook)
# ---------------------------------------------------------------------------

SGPA_COLS = [f"SGPA_Sem{i}" for i in range(1, 9)]

CORE_NUMERIC = [
    "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "Publications", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore", "ExtraCurricular",
]

DISTRIBUTION_COLS = [
    "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore",
]

BOXPLOT_COLS = [
    "CGPA", "Internships", "Projects", "Workshops", "Certifications",
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore",
]

IMPUTE_COLS = [
    "Workshops", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore",
]

STANDARDIZE_COLS = [
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore",
]

CATEGORICAL_RATE_COLS = [
    "Gender", "CollegeTier", "CGPA_Tier", "Stream", "HistoryOfBacklogs",
    "Hostel",
]

TARGET = "PlacementStatus"

# Minimum schema an uploaded file must have for the EDA pages to work.
REQUIRED_COLS = set(CORE_NUMERIC + [TARGET])

# group, role, note — for the Analyse Features registry
COLUMN_META = {
    "StudentID": ("Identity", "identifier", "unique record key, not a feature"),
    "Gender": ("Demographics", "feature", "Male / Female"),
    "City": ("Demographics", "feature", "home city of the student"),
    "CollegeTier": ("Demographics", "feature", "Tier1 / Tier2 / Tier3 institution band"),
    "Stream": ("Demographics", "feature", "degree branch — CS, IT, ECE, EE, Mechanical, Civil"),
    "Specialisation": ("Demographics", "feature", "focus area within the stream"),
    "Hostel": ("Demographics", "feature", "whether the student lives on campus"),
    "HistoryOfBacklogs": ("Academic", "feature", "any failed semester on record"),
    "CGPA": ("Academic", "feature", "cumulative grade point average, 0–10"),
    "AttendancePercent": ("Academic", "feature", "overall attendance, percent"),
    "CGPA_Tier": ("Academic", "derived", "Low / Mid / High banding of CGPA"),
    "Internships": ("Experience", "feature", "count of internships completed"),
    "Projects": ("Experience", "feature", "count of academic / personal projects"),
    "Workshops": ("Experience", "feature", "workshops attended — has missing values"),
    "Certifications": ("Experience", "feature", "external certifications earned"),
    "Publications": ("Experience", "feature", "papers published"),
    "ExtraCurricular": ("Experience", "feature", "count of extra-curricular activities"),
    "AptitudeTestScore": ("Skill scores", "feature", "aptitude test, 0–100 — has missing values"),
    "SoftSkillsRating": ("Skill scores", "feature", "soft-skills rating, 0–5 — has missing values"),
    "CodingTestScore": ("Skill scores", "feature", "coding test, 0–100 — has missing values"),
    "MockInterviewScore": ("Skill scores", "feature", "mock interview, 0–100 — has missing values"),
    "PlacementStatus": ("Outcome", "target", "1 = Placed, 0 = Not placed"),
    "IsAnomaly": ("Outcome", "flag", "1 marks synthetic / anomalous records"),
}

for _col in SGPA_COLS:
    COLUMN_META[_col] = ("Academic", "feature", f"semester {_col[-1]} grade point average")


# ---------------------------------------------------------------------------
# Loading + caching
# ---------------------------------------------------------------------------

_df_cache = {}
_bundle_cache = {}


def load_dataframe(path):
    """Read (and cache) a CSV/Excel dataset, keyed by path + mtime."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _df_cache:
        _df_cache.clear()  # keep memory bounded — only one active dataset
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        _df_cache[key] = df
    return _df_cache[key]


def schema_ok(df):
    return REQUIRED_COLS.issubset(set(df.columns))


def get_bundle(path):
    """All EDA artifacts for the dataset at `path`, computed once and cached.

    Keyed by (path, mtime) like the DataFrame cache — keying by id(df) is
    unsafe because a garbage-collected frame's id can be recycled."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _bundle_cache:
        _bundle_cache.clear()  # only one active dataset — keep memory bounded
        _bundle_cache[key] = _build_bundle(load_dataframe(path))
    return _bundle_cache[key]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(value, nd=2):
    return round(float(value), nd)


def _i(value):
    return int(value)


def _histogram(series, max_bins=24):
    """Equal-width histogram; integer-aligned bins for low-cardinality counts."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return {"labels": [], "counts": [], "smooth": [], "mean": 0, "std": 0, "min": 0, "max": 0}
    lo, hi = float(s.min()), float(s.max())
    nunique = int(s.nunique())
    if nunique <= 12:
        edges = np.arange(lo, hi + 1.5, 1.0) - 0.5
    else:
        edges = np.linspace(lo, hi, max_bins + 1)
    counts, edges = np.histogram(s, bins=edges)
    counts = counts.astype(float)
    # lightly smoothed overlay line, like the KDE in the notebook
    smooth = np.convolve(counts, [0.25, 0.5, 0.25], mode="same")
    labels = [f"{_f(edges[k], 1)}–{_f(edges[k + 1], 1)}" for k in range(len(counts))]
    return {
        "labels": labels,
        "counts": [int(c) for c in counts],
        "smooth": [_f(v, 1) for v in smooth],
        "mean": _f(s.mean()),
        "std": _f(s.std()),
        "min": _f(lo),
        "max": _f(hi),
    }


def _box_stats(series):
    """Box-and-whisker stats with 1.5·IQR fences (whiskers at innermost points).
    Returns None when the group has no values (e.g. a one-class upload)."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return None
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = s[(s >= lo_fence) & (s <= hi_fence)]
    return {
        "min": _f(inside.min()),
        "q1": _f(q1),
        "median": _f(med),
        "q3": _f(q3),
        "max": _f(inside.max()),
        "mean": _f(s.mean()),
        "outliers": _i(((s < lo_fence) | (s > hi_fence)).sum()),
    }


def _heat_color(value):
    """Sequential amber cell color for the correlation heatmap (all
    correlations in this dataset are >= 0, so a single-hue scale reads
    cleaner than a diverging one). Matched to the UI palette."""
    if value is None:
        return "#1B1F1C"
    v = max(0.0, min(1.0, value))
    base = (27, 31, 28)                                   # raised surface
    top = (217, 166, 63)                                  # flat amber accent
    t = v ** 0.7                                          # boost mid-tones
    rgb = tuple(round(b + (c - b) * t) for b, c in zip(base, top))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _clean_categories(df, col):
    """Value counts/rates for a categorical column, dropping anomalous rows
    (the dataset flags a handful of records with numeric 0 placeholders)."""
    sub = df[df[col].apply(lambda v: isinstance(v, str))]
    grouped = sub.groupby(col)[TARGET].agg(["count", "mean"])
    grouped = grouped.sort_values("mean", ascending=False)
    return [
        {"label": str(idx), "count": _i(row["count"]), "rate": _f(row["mean"] * 100, 1)}
        for idx, row in grouped.iterrows()
    ]


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

def _build_bundle(df):
    # An uploaded file that doesn't match the placement schema gets a bare
    # bundle — the stage templates render a schema-mismatch notice for it.
    if not schema_ok(df):
        return {"schema_ok": False}

    # The bundled file ships one corrupt sentinel row (StudentID 0) whose
    # values are the per-column missing counts — impossible on every scale
    # (e.g. Workshops = 4488 on a 0-4 scale). Drop it before any analysis.
    dropped_rows = 0
    if "StudentID" in df.columns:
        dropped_rows = _i((df["StudentID"] == 0).sum())
        df = df[df["StudentID"] != 0]

    bundle = {
        "schema_ok": schema_ok(df),
        "n_rows": _i(df.shape[0]),
        "n_cols": _i(df.shape[1]),
        "dropped_rows": dropped_rows,
    }

    # -- overview (home page) ------------------------------------------------
    placed = df[TARGET].value_counts()
    bundle["overview"] = {
        "rows": _i(df.shape[0]),
        "columns": _i(df.shape[1]),
        "placed": _i(placed.get(1, 0)),
        "not_placed": _i(placed.get(0, 0)),
        "placement_rate": _f(df[TARGET].mean() * 100, 1),
        "avg_cgpa": _f(df["CGPA"].mean()),
        "avg_attendance": _f(df["AttendancePercent"].mean(), 1),
        "avg_aptitude": _f(df["AptitudeTestScore"].mean(), 1),
        "anomalies": _i(df["IsAnomaly"].sum()) if "IsAnomaly" in df else 0,
        "missing_total": _i(df.isna().sum().sum()),
        "completeness": _f(100 * (1 - df.isna().sum().sum() / df.size), 1),
    }

    # -- feature registry (stage 02) ------------------------------------------
    registry = []
    for col in df.columns:
        group, role, note = COLUMN_META.get(col, ("Other", "feature", ""))
        series = df[col]
        samples = [str(v) for v in series.dropna().unique()[:3]]
        registry.append({
            "name": col,
            "group": group,
            "role": role,
            "note": note,
            "dtype": str(series.dtype),
            "non_null": _i(series.notna().sum()),
            "missing": _i(series.isna().sum()),
            "non_null_pct": _f(series.notna().mean() * 100, 1),
            "unique": _i(series.nunique()),
            "samples": samples,
        })
    bundle["features"] = {
        "registry": registry,
        "groups": list(dict.fromkeys(c["group"] for c in registry)),
        "n_numeric": _i(df.select_dtypes(include="number").shape[1]),
        "n_categorical": _i(df.select_dtypes(exclude="number").shape[1]),
    }

    # -- descriptive statistics (stage 03) ------------------------------------
    describe_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in ("StudentID", TARGET, "IsAnomaly")
    ]
    desc = df[describe_cols].describe()
    bundle["descriptive"] = {
        "columns": describe_cols,
        "stats": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        "table": {
        col: {
            "count": _i(desc.loc["count", col]),
            "mean": _f(desc.loc["mean", col]),
            "std": _f(desc.loc["std", col]),
            "min": _f(desc.loc["min", col]),
            "25%": _f(desc.loc["25%", col]),
            "50%": _f(desc.loc["50%", col]),
            "75%": _f(desc.loc["75%", col]),
            "max": _f(desc.loc["max", col]),
        }
        for col in describe_cols
        },
    }

    by_status = []
    for col in CORE_NUMERIC:
        if col not in df:
            continue
        means = df.groupby(TARGET)[col].mean()
        not_placed = float(means.get(0, np.nan))
        is_placed = float(means.get(1, np.nan))
        if np.isnan(not_placed) or np.isnan(is_placed):
            # one-class upload: show the only group's mean on both sides
            only = float(df[col].mean())
            not_placed = is_placed = only
        by_status.append({
            "name": col,
            "not_placed": _f(not_placed),
            "placed": _f(is_placed),
            "delta": _f(is_placed - not_placed),
        })
    bundle["by_status"] = by_status

    # -- missing values (stage 04) ---------------------------------------------
    missing_counts = df.isna().sum()
    affected = missing_counts[missing_counts > 0]
    bundle["missing"] = {
        "total": _i(missing_counts.sum()),
        "total_cells": _i(df.size),
        "completeness": _f(100 * (1 - missing_counts.sum() / df.size), 1),
        "affected": [
            {
                "name": col,
                "count": _i(affected[col]),
                "pct": _f(affected[col] / len(df) * 100, 1),
                "non_null_pct": _f(100 - affected[col] / len(df) * 100, 1),
                "impute_mean": _f(df[col].mean()),
            }
            for col in affected.index
        ],
        "chart_labels": [str(c) for c in affected.index],
        "chart_values": [_i(v) for v in affected.values],
    }

    # -- imputed frame drives every chart below (notebook §6) ------------------
    dfi = df.copy()
    for col in IMPUTE_COLS:
        if col in dfi:
            dfi[col] = dfi[col].fillna(dfi[col].mean())

    # -- distributions (stage 05) ----------------------------------------------
    bundle["histograms"] = {
        col: _histogram(dfi[col]) for col in DISTRIBUTION_COLS if col in dfi
    }

    standardized = {}
    for col in STANDARDIZE_COLS:
        if col not in dfi:
            continue
        z = (dfi[col] - dfi[col].mean()) / dfi[col].std()
        standardized[col] = _histogram(z)
        standardized[col]["mean"] = _f(z.mean(), 3)
        standardized[col]["std"] = _f(z.std(), 3)
    bundle["standardized"] = standardized

    # -- correlation heatmap ----------------------------------------------------
    heat_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in ("StudentID", "IsAnomaly")
    ]
    corr = df[heat_cols].corr()
    matrix = []
    for r in heat_cols:
        row = []
        for c in heat_cols:
            v = corr.loc[r, c]
            v = None if pd.isna(v) else _f(v)
            # ink text once the amber fill is bright enough to carry it
            strong = v is not None and (max(0.0, min(1.0, v)) ** 0.7) >= 0.45
            row.append({"v": v, "color": _heat_color(v), "strong": strong})
        matrix.append(row)
    bundle["heatmap"] = {"labels": heat_cols, "matrix": matrix}

    # -- influence on the target -------------------------------------------------
    infl = (
        dfi[[c for c in CORE_NUMERIC if c in dfi] + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
    )
    bundle["influence"] = {
        "labels": [str(c) for c in infl.index],
        "values": [0 if pd.isna(v) else _f(v, 3) for v in infl.values],
    }
    bundle["top_drivers"] = [
        {"name": str(name), "value": 0 if pd.isna(val) else _f(val, 2)}
        for name, val in infl.head(5).items()
    ]

    # -- boxplots split by placement status --------------------------------------
    boxes = {}
    for col in BOXPLOT_COLS:
        if col not in dfi:
            continue
        b0 = _box_stats(dfi.loc[dfi[TARGET] == 0, col])
        b1 = _box_stats(dfi.loc[dfi[TARGET] == 1, col])
        present = [b for b in (b0, b1) if b]
        if not present:
            continue
        lo = min(b["min"] for b in present)
        hi = max(b["max"] for b in present)
        span = (hi - lo) or 1.0

        def _norm(b):
            return {k: round((b[k] - lo) / span, 4) for k in ("min", "q1", "median", "q3", "max", "mean")}

        boxes[col] = {
            "not_placed": {**b0, "n": _norm(b0)} if b0 else None,
            "placed": {**b1, "n": _norm(b1)} if b1 else None,
            "lo": _f(lo),
            "hi": _f(hi),
        }
    bundle["boxes"] = boxes

    # -- categorical placement rates + gender split ------------------------------
    bundle["categories"] = {
        col: _clean_categories(df, col) for col in CATEGORICAL_RATE_COLS if col in df
    }

    gender_rows = df[df["Gender"].apply(lambda v: isinstance(v, str))]
    g = (
        gender_rows.groupby(["Gender", TARGET]).size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    bundle["gender_split"] = {
        "labels": [str(x) for x in g.index],
        "not_placed": [_i(v) for v in g[0].values],
        "placed": [_i(v) for v in g[1].values],
    }

    return bundle
