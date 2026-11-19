from pathlib import Path

import pandas as pd

from text_processing import clean_token_group_id


BASE_DIR = Path(__file__).resolve().parent
DATASET_CANDIDATES = tuple(
    BASE_DIR / name for name in ("emails.csv", "emails.xls", "emails.xlsx")
)


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    try:
        # Some datasets are CSV files saved with an .xls extension.
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    if path is not None:
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        return _read_dataset(dataset_path)

    for dataset_path in DATASET_CANDIDATES:
        if dataset_path.exists():
            return _read_dataset(dataset_path)

    expected = ", ".join(path.name for path in DATASET_CANDIDATES)
    raise FileNotFoundError(f"No dataset found. Expected one of: {expected}")


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"text", "spam"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    normalized = df[["text", "spam"]].copy()
    missing_text = normalized["text"].isna()
    missing_labels = normalized["spam"].isna()
    invalid_rows = missing_text | missing_labels
    if invalid_rows.any():
        rows = normalized.index[invalid_rows].tolist()
        raise ValueError(
            "Dataset contains records with missing text or labels "
            f"(record indices: {rows}). No records were removed."
        )

    normalized["text"] = normalized["text"].astype(str)
    blank_text = normalized["text"].str.strip().eq("")
    if blank_text.any():
        rows = normalized.index[blank_text].tolist()
        raise ValueError(
            "Dataset contains empty or whitespace-only messages "
            f"(record indices: {rows}). No records were removed."
        )

    numeric_labels = pd.to_numeric(normalized["spam"], errors="coerce")
    invalid_labels = numeric_labels.isna() | ~numeric_labels.isin([0, 1])
    if invalid_labels.any():
        rows = normalized.index[invalid_labels].tolist()
        values = normalized.loc[invalid_labels, "spam"].tolist()
        raise ValueError(
            "Dataset contains labels other than 0 (not spam) or 1 (spam) "
            f"at record indices {rows}: {values}. No records were removed."
        )

    normalized["spam"] = numeric_labels.astype(int)
    return normalized.reset_index(drop=True)


def add_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Group messages with identical cleaned tokens and reject label conflicts."""
    grouped = df.copy()
    grouped["group_id"] = grouped["text"].map(clean_token_group_id)

    label_counts = grouped.groupby("group_id")["spam"].nunique()
    conflicting_groups = label_counts[label_counts > 1].index.tolist()
    if conflicting_groups:
        details = []
        for group_id in conflicting_groups:
            group_rows = grouped.loc[grouped["group_id"] == group_id]
            labels = sorted(int(label) for label in group_rows["spam"].unique())
            record_indices = group_rows.index.tolist()
            details.append(
                f"{group_id}: labels={labels}, record_indices={record_indices}"
            )
        raise ValueError(
            "Messages with identical cleaned-token sequences have conflicting "
            "spam labels. No records were removed or relabeled. Resolve these "
            "labels explicitly before training:\n- "
            + "\n- ".join(details)
        )

    return grouped
