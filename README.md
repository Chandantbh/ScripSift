# ScripSift — Email Spam Detection System

ScripSift is an NLP-based machine learning project that classifies email text as `Spam` or `Not Spam`.

This project is built as a resume-ready personal growth project. It goes beyond a basic notebook by including model comparison, saved evaluation metrics, command-line prediction, a Streamlit web app, and a project report.

## Highlights

- Built with Python, scikit-learn, and Streamlit.
- Compares multiple machine learning pipelines.
- Preserves `not`, `no`, `never`, and `without` during English stop-word filtering.
- Groups messages with identical cleaned-token sequences before splitting.
- Selects the best model using validation spam F1-score.
- Reports final metrics from a grouped independent test set.
- Saves the trained model for reuse.
- Includes terminal prediction, model-based feature explanations, and a refined
  theme-aware Streamlit dashboard.
- Keeps the original notebook as proof of exploration.

## Grouped Evaluation Results

| Metric | Value |
| --- | ---: |
| Selected model | Naive Bayes CountVectorizer |
| Records | 5,728 |
| Unique cleaned-token groups | 5,511 |
| Training rows | 3,665 |
| Validation rows | 917 |
| Independent test rows | 1,146 |
| Independent test accuracy | 99.56% (0.9956) |
| Spam precision | 98.91% (0.9891) |
| Spam recall | 99.27% (0.9927) |
| Spam F1-score | 99.09% (0.9909) |
| ROC-AUC | 99.96% (0.9996) |
| Average precision | 99.87% (0.9987) |
| False positives | 3 |
| False negatives | 2 |

The selected pipeline was chosen using validation spam F1, refitted on the
combined training and validation data, and evaluated once on the independent
grouped test set. The confusion matrix (actual rows and predicted columns,
ordered Not Spam then Spam) is:

```text
[[869, 3],
 [  2, 272]]
```

All three candidates were compared using **validation** metrics. Selection used
spam F1-score only:

| Model | Validation spam F1-score |
| --- | ---: |
| Naive Bayes CountVectorizer | 0.9909 |
| Logistic Regression TF-IDF | 0.9776 |
| ComplementNB TF-IDF | 0.9647 |

## Split and Grouping Method

- Preprocessing removes standard English stop words while preserving the
  negation words `not`, `no`, `never`, and `without`.
- A SHA-256 group ID is generated from each message's JSON-serialized,
  negation-aware `clean_tokens` sequence. Messages with identical cleaned-token
  sequences, including exact duplicate rows, stay together in one split.
- Similarity thresholds are not used to merge messages.
- Group labels are checked before splitting. Conflicting labels produce a
  clear error with group IDs and record indices; records are not silently
  removed or relabeled.
- Two-stage `StratifiedGroupKFold` creates approximately 64% training, 16%
  validation, and 20% final test data using split seeds 42 and 43. The
  train, validation, and test group sets have zero overlap.
- Each candidate is fitted on training data and compared on validation data.
  The selected pipeline is then refitted on training plus validation data and
  evaluated on the untouched grouped test set.
- The current evaluation contains 5,728 records and 5,511 unique
  cleaned-token groups: 3,665 training rows, 917 validation rows, and 1,146
  independent test rows. Split details are written to `model_metadata.json`;
  per-model validation results and the selected model's final test results are
  written to `model_metrics.json`.

## Explainable AI and Dashboard

The scanner displays feature-level contributions from the fitted model and
vectorizer. For the selected Multinomial Naive Bayes model, these are based on
the Spam-versus-Not-Spam feature log-likelihood differences; Logistic Regression
uses its fitted feature coefficients. Explanations use the model's actual
learned feature names, so retained negation terms can appear in the evidence.
These contributions describe the model's decision and are not proof that a
message is malicious or safe.

The Streamlit dashboard uses a restrained, theme-aware obsidian/slate visual
style in Dark and Light modes, with clean surfaces, hairline borders, and
subtle semantic colors. It presents predictions, probabilities, feature
explanations, dataset analytics, and grouped evaluation results.

## Project Structure

```text
email detection/
|-- app.py
|-- data_utils.py
|-- email_spam_detection (1).ipynb
|-- email_spam_detection_professional.ipynb
|-- emails.xls
|-- model_metadata.json
|-- model_metrics.json
|-- predict.py
|-- PROJECT_REPORT.md
|-- README.md
|-- requirements.txt
|-- spam_detection_model.joblib
|-- text_processing.py
`-- train_model.py
```

## Files

| File | Purpose |
| --- | --- |
| `train_model.py` | Trains and compares multiple models. |
| `predict.py` | Predicts spam/not spam from the terminal. |
| `app.py` | Runs the Streamlit web app. |
| `data_utils.py` | Loads and validates the dataset. |
| `text_processing.py` | Contains reusable text cleaning logic. |
| `PROJECT_REPORT.md` | Explains the project professionally. |
| `model_metrics.json` | Stores model comparison results. |
| `model_metadata.json` | Stores selected model and dataset summary. |
| `spam_detection_model.joblib` | Saved trained model. |
| `email_spam_detection (1).ipynb` | Original exploratory notebook. |
| `email_spam_detection_professional.ipynb` | Cleaner presentation notebook. |

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train The Model

```bash
python train_model.py
```

This generates:

```text
spam_detection_model.joblib
model_metrics.json
model_metadata.json
```

## Predict From Terminal

```bash
python predict.py
```

Then paste an email message when prompted.

Example output:

```text
Prediction: Spam
Confidence: 52.13%
Spam probability: 52.13%
Not spam probability: 47.87%
```

## Run The Web App

```bash
streamlit run app.py
```

The web app lets you paste an email message and view:

- Final prediction.
- Confidence score.
- Spam probability.
- Not-spam probability.
- Model-based feature contributions when the selected model supports them.
- Dataset and grouped evaluation analytics.

## Dataset Format

The dataset must contain:

| Column | Description |
| --- | --- |
| `text` | Email/message content. |
| `spam` | Label where `1` means spam and `0` means not spam. |

Training retains all records. Missing or blank text and labels outside `0` or
`1` stop training with an explicit error. Duplicate rows are retained and
grouped so they cannot cross dataset splits.

## Why This Project Is Resume-Worthy

- It is not just a notebook.
- It has a real Streamlit web app.
- It compares multiple machine learning models.
- It saves the trained model and evaluation metrics.
- It uses spam F1-score, not only accuracy.
- It includes a README and project report.
- It has a clean training and prediction workflow.

## Should You Delete The Original Notebook?

No. Keep `email_spam_detection (1).ipynb`.

The original notebook shows your learning process and experimentation. The professional scripts are better for running the final project, but the notebook is useful as proof of work and for explaining how the project developed.

Use `email_spam_detection_professional.ipynb` when you want a cleaner notebook for presentation.

## Limitations

- Very short messages may not contain enough useful words for confident prediction.
- The model depends on the quality and variety of the training data.
- Real-world spam and phishing messages change over time.

## Future Improvements

- Add a TensorFlow LSTM model for deeper comparison.
- Add email metadata features such as sender domain and number of links.
- Deploy the Streamlit app online.
- Train on more modern spam and phishing examples.

## Resume Bullet

Built an NLP-based email spam detection system with negation-aware text
preprocessing, validation-F1 model selection, and a grouped independent test
set; achieved 99.56% test accuracy and added model-based feature explanations
in a theme-aware Streamlit dashboard.

