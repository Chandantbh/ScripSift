# ScripSift — Email Spam Detection Project Report

## Problem Statement

ScripSift classifies email as `Spam` or `Not Spam` using natural language processing and machine learning to help address inbox clutter, fraud, and phishing risks.

## Dataset

The project uses `emails.xls`. The required columns are:

- `text` - the email content.
- `spam` - the label, where `1` means spam and `0` means not spam.

Training retains all records. Missing or blank text and labels outside `0` or
`1` stop training with an explicit error. Duplicate rows are retained and
grouped so they cannot cross dataset splits.

## Negation-Aware Text Preprocessing

Text is lowercased, punctuation is stripped, and standard English stop words
are removed while preserving `not`, `no`, `never`, and `without`. The custom
token cleaner is used by the CountVectorizer pipeline, and both TF-IDF
pipelines use the matching custom stop-word list. This prevents those
negations from being discarded during feature extraction.

## Methodology

The project now compares multiple machine learning pipelines:

- Naive Bayes with `CountVectorizer`.
- Complement Naive Bayes with `TF-IDF`.
- Logistic Regression with `TF-IDF`.

The final model is selected using validation spam F1-score, because spam
detection should care about more than accuracy. A high spam F1-score means the
model is doing a better job balancing false alarms and missed spam.

## Grouped Training and Evaluation

The evaluation prevents messages with identical cleaned-token sequences from
appearing in different splits:

1. The negation-aware `clean_tokens(text)` sequence is serialized consistently
   and hashed with SHA-256 to create a group ID. Approximate-similarity
   thresholds are not used.
2. Before splitting, the trainer checks whether a group contains both spam
   labels. If it does, training stops and reports the group ID, labels, and
   record indices; no records are removed or relabeled.
3. Two-stage `StratifiedGroupKFold` creates training, validation, and final
   test data. The split seeds are 42 and 43. The resulting proportions are
   approximately 64%, 16%, and 20%.
4. All three candidate pipelines are fitted on training data only. Spam
   F1-score on validation data is the sole model-selection metric.
5. The selected pipeline is retrained using training plus validation data,
   then evaluated once on the untouched grouped test set.

The current run retained all 5,728 records and identified 5,511 unique
cleaned-token groups. There are zero conflicting-label groups and zero
group-ID overlap between splits:

| Split | Rows | Unique groups | Not Spam | Spam |
| --- | ---: | ---: | ---: | ---: |
| Training | 3,665 | 3,527 | 2,790 | 875 |
| Validation | 917 | 882 | 698 | 219 |
| Independent test | 1,146 | 1,102 | 872 | 274 |

Validation model comparison (selection uses spam F1 only):

| Model | Validation accuracy | Validation spam F1 |
| --- | ---: | ---: |
| Naive Bayes CountVectorizer | 99.56% | 0.9909 |
| Logistic Regression TF-IDF | 98.91% | 0.9776 |
| ComplementNB TF-IDF | 98.36% | 0.9647 |

Naive Bayes CountVectorizer was selected using validation spam F1. After
refitting on training plus validation data, its grouped independent test
results were:

- Accuracy: **99.56% (0.9956)**
- Spam precision: **98.91% (0.9891)**
- Spam recall: **99.27% (0.9927)**
- Spam F1-score: **99.09% (0.9909)**
- ROC-AUC: **99.96% (0.9996)**
- Average precision: **99.87% (0.9987)**
- Confusion matrix (actual rows, predicted columns; Not Spam then Spam):
  `[[869, 3], [2, 272]]`
- False positives: **3**; false negatives: **2**.

## Explainable AI and Dashboard

`predict.py` returns feature-level explanations from the fitted model and its
vectorizer. For the selected Multinomial Naive Bayes pipeline, the explanation
uses each present feature's Spam-versus-Not-Spam log-likelihood difference
weighted by that feature's value in the message. The Logistic Regression
pipeline uses its learned coefficients. Explanations use the vectorizer's
learned feature names, including retained negation features where present, and
are evidence of model behavior rather than proof of message intent.

The Streamlit dashboard presents predictions, probabilities, explanations,
and dataset/evaluation analytics in a refined, theme-aware minimalist UI.
Its visual style uses monochrome obsidian/slate surfaces, fine borders, clear
typography, and restrained semantic accents in both Light and Dark themes.

## Why Not Only TensorFlow?

The referenced GeeksforGeeks article uses TensorFlow with an Embedding layer and LSTM. That is a good learning approach, but deep learning is not always better for small or medium tabular text datasets. Traditional NLP models such as Naive Bayes and Logistic Regression are faster, easier to explain, and often perform very well for spam detection.

For a professional project, the best approach is to show a baseline model first, then optionally compare it with a TensorFlow model.

## Current Deliverables

- Training script: `train_model.py`
- Prediction script: `predict.py`
- Web app: `app.py`
- Saved model: `spam_detection_model.joblib`
- Metrics output: `model_metrics.json`
- Metadata output: `model_metadata.json`
- Original notebook: `email_spam_detection (1).ipynb`

## Limitations

- Very short messages can be difficult to classify accurately.
- The model depends on the quality and variety of the training dataset.
- Real-world phishing emails may use new wording that the model has not seen.

## Future Improvements

- Add a TensorFlow LSTM model for comparison.
- Add visual charts for label distribution and confusion matrix.
- Add email header features such as sender domain, links, and suspicious attachments.
- Deploy the Streamlit app online.
- Collect more modern spam/phishing examples for better real-world performance.
