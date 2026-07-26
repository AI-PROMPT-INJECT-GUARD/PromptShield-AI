# Agents.md

## Purpose

This repository currently contains the data preparation, exploration, training, and local inference assets for a prompt-injection detection project called PromptShield AI.

Despite the `README.md` mentioning FastAPI and Streamlit, there is no web app code in the repository at the time of this scan. The implemented project is a Python ML workflow centered on CSV datasets and a DistilBERT classifier.

## Current State

- Primary language: Python
- Core workload: dataset download/filtering/merging, model training, local CLI prediction
- Model family: `distilbert-base-uncased`
- Main saved artifact referenced by code: `./saved_model`
- Large archived artifact present: `saved_model.zip`
- Dependency manifest: `requirements.txt` exists but is empty

## Repository Layout

### Root

- `README.md`: short project overview and planned features
- `requirements.txt`: present but empty
- `download_dataset.py`: downloads `yahma/alpaca-cleaned` and prints a sample
- `saved_model.zip`: archived model artifact

### `dataset/`

- `download_dataset.py`: downloads `neuralchemy/Prompt-injection-dataset` (`core`) into `dataset/raw/`
- `explore_dataset.py`: prints schema and label distribution for raw splits
- `prompt_injection_only.csv`: data artifact already present
- `safe_prompts.csv`: generated safe-prompt dataset from Alpaca

#### `dataset/raw/`

Raw splits from `neuralchemy/Prompt-injection-dataset`:

- `train.csv`
- `validation.csv`
- `test.csv`

These files contain columns such as:

- `text`
- `label`
- `category`
- `source`
- `severity`
- `group_id`
- `augmented`
- `tags`

#### `dataset/processed/`

Processed versions of the raw data:

- `train.csv`
- `validation.csv`
- `test.csv`
- `train_balanced.csv`

`ml/preprocess_dataset.py` reduces raw files to `text` and `label`, removes duplicates, and writes these outputs.

`train_balanced.csv` is created by merging processed training data with sampled safe prompts from Alpaca.

#### `dataset/shomi_dataset/`

Downloaded from `Shomi28/prompt-injection-dataset`:

- `train.csv`
- `validation.csv`
- `test.csv`

Observed schema:

- `text`
- `label`
- `label_name`

This dataset is the one used directly by `ml/train_model.py`.

#### `dataset/pi_dataset/`

Downloaded from `deepset/prompt-injections`:

- `train.csv`
- `clean_prompt_injection.csv`
- `final_prompt_injection.csv`
- `review_prompt_injections.csv`

These files are used for filtering and reviewing positive prompt-injection examples.

### `ml/`

#### Training and inference

- `train_model.py`
  - reads `dataset/shomi_dataset/{train,validation,test}.csv`
  - tokenizes `text`
  - trains `DistilBertForSequenceClassification`
  - evaluates on validation data through Hugging Face `Trainer`
  - saves model and tokenizer to `./saved_model`

- `predict.py`
  - loads tokenizer and model from `./saved_model`
  - first applies a rule-based pattern list
  - if no rule matches, runs model inference
  - expects interactive terminal input via `input()`

#### Dataset acquisition and preparation

- `download_large_pi_dataset.py`: downloads `Shomi28/prompt-injection-dataset`
- `download_pi_dataset.py`: downloads `deepset/prompt-injections`
- `prepare_safe_dataset.py`: converts `yahma/alpaca-cleaned` into label `0` prompts
- `merge_datasets.py`: samples 2500 safe prompts and merges them into processed training data

#### Filtering and review helpers

- `filter_injection_dataset.py`: keyword-filters positive examples from `dataset/pi_dataset/train.csv`
- `filter_deepset_dataset.py`: keeps likely true injection patterns and removes roleplay-like prompts
- `export_positive_examples.py`: exports positive examples for manual review with an empty `keep` column

#### Exploration scripts

- `explore_shomi_dataset.py`
- `explore_pi_dataset.py`
- `explore_injection_labels.py`

These are print-only inspection scripts used during data analysis.

## Effective Workflow

The current repo supports this practical flow:

1. Download one or more source datasets.
2. Inspect or filter them.
3. Build processed or merged CSVs.
4. Train a DistilBERT binary classifier.
5. Run local CLI inference with a hybrid rule-based and model-based detector.

## Important Couplings

- `ml/train_model.py` is coupled to `dataset/shomi_dataset/`, not `dataset/processed/`.
- `ml/predict.py` requires a directory named `saved_model/`, while the root currently exposes `saved_model.zip`.
- `ml/merge_datasets.py` depends on:
  - `dataset/processed/train.csv`
  - `dataset/safe_prompts.csv`
- Several scripts download from Hugging Face and therefore require network access at runtime.

## Data and Model Assumptions

- Label convention is binary:
  - `0` = safe
  - `1` = prompt injection
- Training code assumes a `text` column exists in the selected dataset.
- Tokenization is fixed to `max_length=128`.
- Training code uses validation data in the `Trainer`, but the loaded test split is not separately consumed after load.

## Gaps and Risks

- `requirements.txt` is empty, so the environment cannot be reproduced from the repo alone.
- `README.md` describes planned app capabilities that are not implemented in this codebase.
- `ml/predict.py` contains mojibake in printed emoji output, which suggests an encoding issue in the file.
- The repository mixes exploratory scripts, generated CSV artifacts, and training code without a single orchestrating entry point.
- There is no test suite.
- There is no documented inference API or web interface in the current code.

## Suggested Dependency Baseline

Based on imports present in the code, the project depends at least on:

- `pandas`
- `numpy`
- `scikit-learn`
- `datasets`
- `transformers`
- `torch`

## Practical Notes For Future Agents

- Treat this repo as an ML experiment workspace, not as a finished product application.
- Prefer documenting actual behavior from code over the higher-level claims in `README.md`.
- Before changing training or inference behavior, verify which dataset family is intended:
  - `dataset/raw`
  - `dataset/processed`
  - `dataset/shomi_dataset`
  - `dataset/pi_dataset`
- If you need reproducibility, start by creating a real `requirements.txt` and documenting a canonical training path.
- If you need deployable behavior, the missing pieces are an extracted `saved_model/` directory, dependency pinning, and an actual service or UI layer.

## Files Most Likely To Matter First

- `ml/train_model.py`
- `ml/predict.py`
- `ml/preprocess_dataset.py`
- `ml/merge_datasets.py`
- `dataset/download_dataset.py`
- `README.md`
