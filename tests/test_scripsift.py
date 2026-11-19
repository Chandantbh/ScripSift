import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import predict
from data_utils import add_group_ids, load_dataset
from text_processing import clean_tokens
from train_model import BASE_DIR as TRAIN_BASE_DIR
from train_model import (
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH as TRAIN_MODEL_PATH,
    create_grouped_splits,
)


class ScripSiftTests(unittest.TestCase):
    def test_clean_tokens_preserves_negations_and_removes_stop_words(self):
        tokens = clean_tokens("This is not good, no one should never go without the.")

        for negation in ("not", "no", "never", "without"):
            with self.subTest(negation=negation):
                self.assertIn(negation, tokens)
        for stop_word in ("this", "is", "the", "should"):
            with self.subTest(stop_word=stop_word):
                self.assertNotIn(stop_word, tokens)

    def test_grouped_splits_keep_duplicate_messages_together(self):
        rows = []
        for label in (0, 1):
            for index in range(20):
                text = f"message class{label} unique{index}"
                rows.append({"text": text, "spam": label})
                if index == 0:
                    rows.append(
                        {"text": f"MESSAGE, CLASS{label}! UNIQUE{index}.", "spam": label}
                    )

        grouped = add_group_ids(pd.DataFrame(rows))
        splits = create_grouped_splits(grouped)

        split_for_index = {
            index: split_name
            for split_name, split in splits.items()
            for index in split.index
        }
        for index in (0, 1, 21, 22):
            duplicate_group = grouped.loc[index, "group_id"]
            duplicate_indices = grouped.index[grouped["group_id"] == duplicate_group]
            self.assertEqual(
                len({split_for_index[duplicate_index] for duplicate_index in duplicate_indices}),
                1,
            )

        split_names = tuple(splits)
        for index, first_name in enumerate(split_names):
            for second_name in split_names[index + 1 :]:
                self.assertTrue(
                    set(splits[first_name]["group_id"]).isdisjoint(
                        splits[second_name]["group_id"]
                    )
                )
        self.assertEqual(set(split_for_index), set(grouped.index))

    def test_invalid_prediction_inputs_do_not_load_or_call_model(self):
        invalid_inputs = (
            None,
            42,
            "",
            " \t\r\n ",
            "bad\x00message",
            "bad\x7fmessage",
            "x" * (predict.MAX_EMAIL_CHARS + 1),
        )

        with patch.object(predict, "load_model") as load_model:
            for email_text in invalid_inputs:
                with self.subTest(email_text=repr(email_text)[:40]):
                    with self.assertRaises(ValueError):
                        predict.validate_email_text(email_text)
                    with self.assertRaises(ValueError):
                        predict.predict_email(email_text)
                    with self.assertRaises(ValueError):
                        predict.predict_email_details(email_text)

            load_model.assert_not_called()

        allowed_whitespace = predict.validate_email_text("hello\tthere\n")
        self.assertEqual(allowed_whitespace, "hello\tthere\n")
        self.assertEqual(
            predict.validate_email_text("x" * predict.MAX_EMAIL_CHARS),
            "x" * predict.MAX_EMAIL_CHARS,
        )

    def test_model_prediction_and_explanation_work_from_external_directory(self):
        original_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as external_directory:
            try:
                os.chdir(external_directory)
                model = predict.load_model()
                classifier = model.named_steps["classifier"]
                feature_index = abs(
                    classifier.feature_log_prob_[1]
                    - classifier.feature_log_prob_[0]
                ).argmax()
                email_text = model.named_steps["vectorizer"].get_feature_names_out()[
                    feature_index
                ]

                details = predict.predict_email_details(str(email_text))
                explanation = predict.explain_prediction(model, str(email_text))
            finally:
                os.chdir(original_directory)

        self.assertIn(details["label"], ("Spam", "Not Spam"))
        self.assertGreaterEqual(details["confidence"], 0)
        self.assertTrue(details["explanation_available"])
        self.assertTrue(explanation["explanation_available"])

    def test_explanations_support_linear_and_multinomial_nb_models(self):
        training_text = (
            "free cash prize",
            "claim free cash",
            "meeting project report",
            "team meeting project",
        )
        labels = (1, 1, 0, 0)

        for classifier in (
            LogisticRegression(random_state=0),
            MultinomialNB(),
        ):
            with self.subTest(classifier=type(classifier).__name__):
                model = Pipeline(
                    (
                        ("vectorizer", CountVectorizer(analyzer=clean_tokens)),
                        ("classifier", classifier),
                    )
                ).fit(training_text, labels)
                result = predict.explain_prediction(
                    model,
                    "free cash meeting project",
                    top_n=2,
                )

                self.assertTrue(result["explanation_available"])
                self.assertEqual(result["explanation_message"], "")
                for key, sign in (("spam_features", 1), ("not_spam_features", -1)):
                    self.assertTrue(result[key])
                    self.assertLessEqual(len(result[key]), 2)
                    for feature in result[key]:
                        self.assertIsInstance(feature["feature"], str)
                        self.assertGreater(sign * feature["contribution"], 0)

        unsupported = predict.explain_prediction(object(), "free cash")
        out_of_vocabulary = predict.explain_prediction(
            model,
            "zzzxxyyqqq",
        )
        self.assertFalse(unsupported["explanation_available"])
        self.assertTrue(unsupported["explanation_message"])
        self.assertFalse(out_of_vocabulary["explanation_available"])
        self.assertTrue(out_of_vocabulary["explanation_message"])

    def test_load_dataset_accepts_an_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "custom.csv"
            pd.DataFrame({"text": ["hello"], "spam": [0]}).to_csv(
                dataset_path,
                index=False,
            )

            loaded = load_dataset(dataset_path)

        self.assertEqual(loaded.loc[0, "text"], "hello")

    def test_training_paths_are_anchored_to_module_directory(self):
        self.assertEqual(TRAIN_BASE_DIR, Path(__file__).resolve().parents[1])
        self.assertEqual(TRAIN_MODEL_PATH, TRAIN_BASE_DIR / "spam_detection_model.joblib")
        self.assertEqual(METRICS_PATH, TRAIN_BASE_DIR / "model_metrics.json")
        self.assertEqual(METADATA_PATH, TRAIN_BASE_DIR / "model_metadata.json")


if __name__ == "__main__":
    unittest.main()
