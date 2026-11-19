from pathlib import Path

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_detection_model.joblib"
MAX_EMAIL_CHARS = 50_000


def validate_email_text(email_text: object) -> str:
    if not isinstance(email_text, str):
        raise ValueError("Email text must be a string.")
    if not email_text.strip():
        raise ValueError("Email text cannot be empty or whitespace-only.")
    if len(email_text) > MAX_EMAIL_CHARS:
        raise ValueError(
            f"Email text must not exceed {MAX_EMAIL_CHARS:,} characters."
        )
    if any(
        (ord(character) < 32 and character not in "\t\n\r")
        or ord(character) == 127
        for character in email_text
    ):
        raise ValueError(
            "Email text contains unsupported control characters. "
            "Tabs and line breaks are allowed."
        )
    return email_text


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` before prediction."
        )
    return joblib.load(MODEL_PATH)


def _unavailable_explanation(message: str) -> dict:
    return {
        "explanation_available": False,
        "explanation_message": message,
        "spam_features": [],
        "not_spam_features": [],
    }


def explain_prediction(model, email_text: str, top_n: int = 5) -> dict:
    """Explain a binary linear text model using its fitted feature contributions."""
    if not isinstance(email_text, str) or not email_text.strip():
        return _unavailable_explanation(
            "Enter a non-empty email message to generate an explanation."
        )
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    named_steps = getattr(model, "named_steps", None)
    if not named_steps:
        return _unavailable_explanation(
            "This model does not expose a fitted vectorizer and classifier pipeline."
        )

    vectorizer = next(
        (
            step
            for step in named_steps.values()
            if callable(getattr(step, "transform", None))
            and callable(getattr(step, "get_feature_names_out", None))
        ),
        None,
    )
    classifier = next(
        (
            step
            for step in named_steps.values()
            if hasattr(step, "classes_")
            and (hasattr(step, "coef_") or isinstance(step, MultinomialNB))
        ),
        None,
    )
    if vectorizer is None or classifier is None:
        return _unavailable_explanation(
            "Explanations are supported for fitted linear classifiers with a "
            "feature-name-aware vectorizer; this model has a different architecture."
        )

    classes = np.asarray(classifier.classes_)
    if classes.ndim != 1 or len(classes) != 2 or set(classes.tolist()) != {0, 1}:
        return _unavailable_explanation(
            "This classifier is not a supported binary model for "
            "Spam/Not Spam explanations."
        )

    feature_names = vectorizer.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_)
        if coefficients.ndim != 2 or coefficients.shape[0] != 1:
            return _unavailable_explanation(
                "This classifier is not a supported binary linear model for "
                "Spam/Not Spam explanations."
            )
        spam_direction = 1.0 if classes[1] == 1 else -1.0
        feature_weights = coefficients[0] * spam_direction
    else:
        feature_log_prob = np.asarray(classifier.feature_log_prob_)
        if feature_log_prob.ndim != 2 or feature_log_prob.shape[0] != 2:
            return _unavailable_explanation(
                "This Multinomial Naive Bayes model does not expose compatible "
                "binary feature probabilities."
            )
        spam_index = int(np.flatnonzero(classes == 1)[0])
        not_spam_index = int(np.flatnonzero(classes == 0)[0])
        feature_weights = (
            feature_log_prob[spam_index] - feature_log_prob[not_spam_index]
        )

    if feature_weights.shape[0] != len(feature_names):
        return _unavailable_explanation(
            "The classifier coefficients do not match the vectorizer features."
        )

    transformed = vectorizer.transform([email_text])
    row = transformed.getrow(0)
    if row.nnz == 0:
        return _unavailable_explanation(
            "No words or phrases from this email are present in the model vocabulary."
        )

    contributions = row.data * feature_weights[row.indices]

    spam_features = []
    not_spam_features = []
    for feature_index, contribution in zip(row.indices, contributions):
        value = float(contribution)
        if not np.isfinite(value) or value == 0:
            continue
        feature = {
            "feature": str(feature_names[feature_index]),
            "contribution": value,
        }
        if value > 0:
            spam_features.append(feature)
        else:
            not_spam_features.append(feature)

    spam_features.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    not_spam_features.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    spam_features = spam_features[:top_n]
    not_spam_features = not_spam_features[:top_n]
    if not spam_features and not not_spam_features:
        return _unavailable_explanation(
            "The email has no non-zero feature contributions in this model."
        )

    return {
        "explanation_available": True,
        "explanation_message": "",
        "spam_features": spam_features,
        "not_spam_features": not_spam_features,
    }


def predict_email_details(email_text: str) -> dict:
    email_text = validate_email_text(email_text)

    model = load_model()
    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or set(classes.tolist()) != {0, 1}:
        raise ValueError(
            "The loaded model must expose Spam (1) and Not Spam (0) classes."
        )

    prediction = int(model.predict([email_text])[0])
    probabilities = model.predict_proba([email_text])[0]
    class_indices = {int(label): index for index, label in enumerate(classes)}
    not_spam_probability = float(probabilities[class_indices[0]] * 100)
    spam_probability = float(probabilities[class_indices[1]] * 100)
    explanation = explain_prediction(model, email_text)

    return {
        "label": "Spam" if prediction == 1 else "Not Spam",
        "prediction": prediction,
        "confidence": spam_probability if prediction == 1 else not_spam_probability,
        "not_spam_probability": not_spam_probability,
        "spam_probability": spam_probability,
        **explanation,
    }


def predict_email(email_text: str) -> tuple[str, float]:
    details = predict_email_details(email_text)
    return details["label"], details["confidence"]


def main() -> None:
    email_text = input("Paste email text: ").strip()
    if not email_text:
        raise ValueError("Email text cannot be empty.")

    details = predict_email_details(email_text)
    print(f"Prediction: {details['label']}")
    print(f"Confidence: {details['confidence']:.2f}%")
    print(f"Spam probability: {details['spam_probability']:.2f}%")
    print(f"Not spam probability: {details['not_spam_probability']:.2f}%")


if __name__ == "__main__":
    main()
