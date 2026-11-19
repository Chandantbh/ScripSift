import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_utils import add_group_ids, load_dataset, normalize_dataset
from text_processing import CUSTOM_STOP_WORDS_LIST, clean_tokens


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_detection_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"
RANDOM_SEED = 42
TEST_SPLIT_SEED = RANDOM_SEED
VALIDATION_SPLIT_SEED = RANDOM_SEED + 1
SPLIT_FOLDS = 5
GROUPING_METHOD = (
    "SHA-256 of the JSON-serialized negation-aware clean_tokens(text) sequence "
    "(retaining not, no, never, and without); "
    "identical cleaned-token sequences stay in one split."
)


def build_models() -> dict[str, Pipeline]:
    return {
        "Naive Bayes CountVectorizer": Pipeline(
            steps=[
                ("vectorizer", CountVectorizer(analyzer=clean_tokens)),
                ("classifier", MultinomialNB()),
            ]
        ),
        "ComplementNB TF-IDF": Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words=CUSTOM_STOP_WORDS_LIST,
                        ngram_range=(1, 2),
                        max_features=30000,
                    ),
                ),
                ("classifier", ComplementNB()),
            ]
        ),
        "Logistic Regression TF-IDF": Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words=CUSTOM_STOP_WORDS_LIST,
                        ngram_range=(1, 2),
                        max_features=30000,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x, y, evaluation_split: str) -> dict:
    predictions = model.predict(x)
    positive_scores = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(
                "The fitted model does not contain the spam class (label 1)."
            )
        positive_scores = model.predict_proba(x)[:, classes.index(1)]
    elif hasattr(model, "decision_function"):
        positive_scores = model.decision_function(x)

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "evaluation_split": evaluation_split,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_spam": float(
            precision_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "recall_spam": float(
            recall_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "f1_spam": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "roc_auc": None,
        "average_precision": None,
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
    }
    if positive_scores is not None and y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, positive_scores))
        metrics["average_precision"] = float(
            average_precision_score(y, positive_scores)
        )
    return metrics


def _next_group_fold(
    x,
    y,
    groups,
    random_seed: int,
):
    splitter = StratifiedGroupKFold(
        n_splits=SPLIT_FOLDS,
        shuffle=True,
        random_state=random_seed,
    )
    return next(splitter.split(x, y, groups))


def create_grouped_splits(df, random_seed: int = RANDOM_SEED):
    """Create reproducible 64/16/20 splits without separating duplicate groups."""
    class_group_counts = df.groupby("spam")["group_id"].nunique()
    if any(class_group_counts.get(label, 0) < SPLIT_FOLDS for label in (0, 1)):
        raise ValueError(
            "Each class needs at least "
            f"{SPLIT_FOLDS} unique cleaned-token groups for grouped splitting."
        )

    train_validation_indices, test_indices = _next_group_fold(
        df["text"],
        df["spam"],
        df["group_id"],
        random_seed,
    )
    train_validation = df.iloc[train_validation_indices].copy()
    test = df.iloc[test_indices].copy()

    train_indices, validation_indices = _next_group_fold(
        train_validation["text"],
        train_validation["spam"],
        train_validation["group_id"],
        random_seed + 1,
    )
    train = train_validation.iloc[train_indices].copy()
    validation = train_validation.iloc[validation_indices].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    for split_name, split in splits.items():
        if set(split["spam"].unique()) != {0, 1}:
            raise ValueError(
                f"The {split_name} split does not contain both spam labels."
            )

    split_names = list(splits)
    for index, first_name in enumerate(split_names):
        first_groups = set(splits[first_name]["group_id"])
        for second_name in split_names[index + 1 :]:
            overlap = first_groups.intersection(splits[second_name]["group_id"])
            if overlap:
                raise RuntimeError(
                    f"Cleaned-token groups overlap between {first_name} and "
                    f"{second_name}: {sorted(overlap)}"
                )

    all_indices = set().union(*(set(split.index) for split in splits.values()))
    if all_indices != set(df.index) or sum(map(len, splits.values())) != len(df):
        raise RuntimeError("The grouped splits did not preserve every input row.")

    return splits


def class_distribution(df) -> dict[str, int]:
    counts = df["spam"].value_counts().to_dict()
    return {
        "not_spam": int(counts.get(0, 0)),
        "spam": int(counts.get(1, 0)),
    }


def split_summary(splits: dict) -> dict:
    return {
        split_name: {
            "rows": int(len(split)),
            "unique_groups": int(split["group_id"].nunique()),
            "class_distribution": class_distribution(split),
        }
        for split_name, split in splits.items()
    }


def main() -> None:
    df = add_group_ids(normalize_dataset(load_dataset()))
    splits = create_grouped_splits(df)
    split_details = split_summary(splits)

    print(f"Rows retained: {len(df)}")
    print(f"Unique cleaned-token groups: {df['group_id'].nunique()}")
    print("Conflicting-label groups: 0")
    print("\nSplit distribution:")
    for split_name, details in split_details.items():
        print(
            f"- {split_name.title()}: {details['rows']} rows, "
            f"{details['unique_groups']} groups, "
            f"not_spam={details['class_distribution']['not_spam']}, "
            f"spam={details['class_distribution']['spam']}"
        )

    validation_metrics = {}
    for model_name, model in build_models().items():
        model.fit(splits["train"]["text"], splits["train"]["spam"])
        validation_metrics[model_name] = evaluate_model(
            model,
            splits["validation"]["text"],
            splits["validation"]["spam"],
            "validation",
        )

    best_model_name = max(
        validation_metrics,
        key=lambda name: validation_metrics[name]["f1_spam"],
    )
    final_model = build_models()[best_model_name]
    train_validation = pd.concat(
        [splits["train"], splits["validation"]],
        axis=0,
    ).sort_index()
    final_model.fit(train_validation["text"], train_validation["spam"])
    final_test_metrics = evaluate_model(
        final_model,
        splits["test"]["text"],
        splits["test"]["spam"],
        "grouped_independent_test",
    )
    validation_metrics[best_model_name]["final_test"] = final_test_metrics

    overall_class_distribution = class_distribution(df)
    metadata = {
        "best_model": best_model_name,
        "selected_model": best_model_name,
        "split_strategy": (
            "Two-stage StratifiedGroupKFold: approximately 20% grouped test "
            "holdout, then approximately 20% of the remaining data for "
            "validation (approximately 64% train, 16% validation, 20% test)."
        ),
        "grouping_method": GROUPING_METHOD,
        "random_seed": RANDOM_SEED,
        "split_seeds": {
            "test": TEST_SPLIT_SEED,
            "validation": VALIDATION_SPLIT_SEED,
        },
        "rows_used": len(df),
        "train_rows": split_details["train"]["rows"],
        "validation_rows": split_details["validation"]["rows"],
        "test_rows": split_details["test"]["rows"],
        "unique_groups": {
            "total": int(df["group_id"].nunique()),
            "train": split_details["train"]["unique_groups"],
            "validation": split_details["validation"]["unique_groups"],
            "test": split_details["test"]["unique_groups"],
        },
        "class_distribution": overall_class_distribution,
        "split_class_distribution": {
            split_name: details["class_distribution"]
            for split_name, details in split_details.items()
        },
        "selection_metric": "validation_f1_spam",
        "model_selection_metric": "validation_f1_spam",
        "final_test_metrics": final_test_metrics,
        "model_file": MODEL_PATH.name,
        "scikit_learn_version": sklearn.__version__,
    }
    evaluation_metadata = {
        "split_strategy": metadata["split_strategy"],
        "grouping_method": GROUPING_METHOD,
        "random_seed": RANDOM_SEED,
        "split_seeds": metadata["split_seeds"],
        "train_rows": metadata["train_rows"],
        "validation_rows": metadata["validation_rows"],
        "test_rows": metadata["test_rows"],
        "unique_groups": metadata["unique_groups"],
        "model_selection_metric": metadata["model_selection_metric"],
        "selected_model": best_model_name,
    }
    for result in validation_metrics.values():
        result["evaluation_metadata"] = evaluation_metadata

    print("\nValidation model comparison (selection uses validation spam F1 only):")
    for model_name, result in validation_metrics.items():
        print(
            f"- {model_name}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"spam_f1={result['f1_spam']:.4f}"
        )
    print(f"\nSelected model from validation: {best_model_name}")
    print("Final grouped independent test results:")
    for metric_name in (
        "accuracy",
        "precision_spam",
        "recall_spam",
        "f1_spam",
        "roc_auc",
        "average_precision",
    ):
        print(f"- {metric_name}: {final_test_metrics[metric_name]}")
    print("Final test confusion matrix:")
    print(final_test_metrics["confusion_matrix"])

    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(validation_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
