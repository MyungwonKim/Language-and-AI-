# Language and AI Group 29 - Personality Prediction with RoBERTa

This project is an assignment for the course **JBC090 Language and AI**. It implements a machine learning pipeline to predict personality types (introvert/extrovert) using RoBERTa models trained on three different data preprocessing strategies.

## Project Overview

The goal of this project is to analyze how self descriptions masking affect the performance of RoBERTa models in personality prediction tasks. We compare three approaches: 
1. **Raw data** - Original posts with self-descriptions
2. **Masked self-descriptions** - Posts where personality labels (e.g., "I'm an introvert", "As an extrovert") are masked
3. **Random masking** - Posts where random words are masked for comparison

## Pipeline Architecture

```
extrovert_introvert.csv 
    ↓
preprocessing_lang_ai.py (data cleaning)
    ↓
regex_exp.py (filter self-description posts)
    ↓
    ├─→ masking. py (mask self-descriptions)
    ├─→ masking_random.py (mask random words)
    └─→ raw data
    ↓
traintest_split.py (split into train/test/val)
    ↓
RobertaBase.py (model training × 3)
    ↓
metrics_eval.py (performance comparison)
```

## Prerequisites

- Python 3.x
- Required libraries: 
  - pandas
  - scikit-learn
  - transformers (Hugging Face)
  - torch
  - numpy
  - re (regex)

## Installation

```bash
pip install pandas scikit-learn transformers torch numpy
```

## Usage

### Step 1: Data Preprocessing

Start with the `extrovert_introvert.csv` file provided by the course. 

```bash
python preprocessing_lang_ai. py
```

This script cleans the raw data and prepares it for further processing.

### Step 2: Filter Self-Description Posts

```bash
python regex_exp.py
```

This filters posts that include self-descriptions like "I'm an introvert" or "I'm an extrovert".  The output serves as the raw dataset. 

### Step 3: Apply Masking Strategies

Generate three different datasets:

**Dataset 1: Raw data** (no masking)
- Use the output from `regex_exp.py` directly

**Dataset 2: Masked self-descriptions**
```bash
python masking.py
```
This masks all self-description phrases in the posts.

**Dataset 3: Random masking**
```bash
python masking_random.py
```
This randomly masks words in the posts for baseline comparison.

### Step 4: Split Data

For each of the three datasets, run: 

```bash
python traintest_split.py
```

This splits each dataset into train, test, and validation CSV files.

**Important:** After splitting, move each set of files (train. csv, test.csv, val. csv) into separate folders to avoid overwriting: 
- `data/raw_data/`
- `data/masked_self/`
- `data/masked_random/`

### Step 5: Train RoBERTa Models

Train three separate models, one for each dataset:

```bash
python RobertaBase.py
```

**Note:** You need to run this script three times, changing the data path each time to point to: 
1. Raw data folder
2. Masked self-description data folder
3. Random masked data folder

The trained models will be saved for later evaluation.

### Step 6: Evaluate and Compare

Run the evaluation script to compare model performances:

```bash
python metrics_eval.py
```

This generates performance metrics (accuracy, F1-score, precision, recall) for all three models.

## File Descriptions

| File | Description |
|------|-------------|
| `preprocessing_lang_ai.py` | Cleans and preprocesses the raw CSV data |
| `regex_exp.py` | Filters posts containing self-description patterns |
| `masking.py` | Masks self-description phrases in posts |
| `masking_random.py` | Randomly masks words in posts |
| `traintest_split.py` | Splits data into train/test/validation sets |
| `RobertaBase.py` | Trains RoBERTa model on the prepared datasets |
| `metrics_eval.py` | Evaluates and compares model performances |
| `baseline. py` | Baseline model implementation |
| `eda. ipynb` | Exploratory data analysis notebook |
| `performance_comparison.py` | Additional performance comparison utilities |

## Project Structure

```
Language-and-AI-/
├── preprocessing_lang_ai.py
├── regex_exp.py
├── masking.py
├── masking_random.py
├── traintest_split.py
├── RobertaBase. py
├── metrics_eval. py
├── baseline.py
├── performance_comparison.py
├── eda.ipynb
├── README.md
└── data/
    ├── raw_data/
    │   ├── train.csv
    │   ├── test.csv
    │   └── val.csv
    ├── masked_self/
    │   ├── train.csv
    │   ├── test.csv
    │   └── val.csv
    └── masked_random/
        ├── train.csv
        ├── test.csv
        └── val.csv
```
