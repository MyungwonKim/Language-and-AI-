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
regex_exp.py (filter self-description posts - 1st pass)
    ↓
chunking.py (slice posts into chunks)
    ↓
regex_exp.py (filter chunks with self-descriptions - 2nd pass)
    ↓
    ├─→ masking.py (mask self-descriptions)
    ├─→ masking_random.py (mask random words)
    └─→ raw data
    ↓
traintest_split.py (split into train/test/val)
    ↓
RobertaBase.py (model training × 3)
    ↓
metrics_eval.py (performance comparison)
    ↓
word_importance.py (analyze model predictions)
```

## TL;DR - Why This Repository Matters

This repository investigates whether RoBERTa models can predict introversion/extroversion from social media text, and **critically examines whether models rely on explicit self-descriptions** (e.g., "I'm an introvert") rather than genuine linguistic patterns.

**Key Contributions:**
- **Methodology for detecting data leakage:** Demonstrates how to identify and mitigate spurious correlations in NLP classification tasks
- **Controlled masking experiments:** Compares targeted masking (self-descriptions) vs. random masking to isolate the effect
- **Complete reproducible pipeline:** From raw Reddit data to trained models with evaluation and interpretability tools
- **Practical insights:** Shows that models may achieve high accuracy by "cheating" rather than learning meaningful patterns

**Why you should care:**
- If you're building personality prediction models, this shows how to avoid common pitfalls
- Provides reusable code for text preprocessing, chunking, and RoBERTa fine-tuning
- Includes interpretability tools to understand what your model actually learned
- Demonstrates proper experimental design with control conditions

## Reproduction Instructions

### System Requirements & Resources

**Hardware Used:**
- **CPU:** 12th Gen Intel(R) Core(TM) i7-12700H @ 2.30 GHz (14 cores:  6 P-cores + 8 E-cores, 20 threads)
- **RAM:** 16.0 GB (15.6 GB usable) DDR4
- **GPU:** NVIDIA A1000 (8GB VRAM) with CUDA 11.8
- **System Type:** 64-bit operating system, x64-based processor

**Software Environment:**
- **Operating System:** Windows 11 (64-bit)
- **Python Version:** 3.10.12
- **CUDA Version:** 11.8

**Minimum Requirements:**
- 8GB RAM (16GB recommended)
- Python 3.8 or higher
- (Optional) NVIDIA GPU with 8GB+ VRAM and CUDA 11.8+

**GPU Training**
- Estimated time: 30 minutes per model

### Dependencies

All dependencies are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

**Core Dependencies (versions tested):**

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.0.3 | Data manipulation and CSV handling |
| `numpy` | 1.24.3 | Numerical computations |
| `scikit-learn` | 1.3.0 | Train/test splitting, baseline models, metrics |
| `torch` | 2.1.0 | PyTorch deep learning framework |
| `transformers` | 4.35.0 | Hugging Face RoBERTa implementation |
| `nltk` | 3.9.2 | Sentence tokenization for chunking |
| `langdetect` | 1.0.9 | Language detection for filtering |
| `ftfy` | 6.1.3 | Fix text encoding issues |
| `tqdm` | 4.67.1 | Progress bars |
| `regex` | 2025.11.3 | Advanced regex patterns |
| `joblib` | 1.3.2 | Serialization |
| `accelerate` | 0.25.0 | (Optional) Faster training |


### Data Acquisition

The dataset `extrovert_introvert.csv` is provided by the course instructor and should be placed in the `data/` directory.  

**Dataset Statistics:**
- **Total posts:** 40,452 (40,451 after deduplication)
- **Introverts (label=0):** 31,370 posts (77.5%)
- **Extroverts (label=1):** 9,082 posts (22.5%)
- **Unique authors:** ~3,000+
- **Source:** Reddit posts from personality-related subreddits

**Expected CSV format:**
```csv
auhtor_ID,post,extrovert
t2_12345,"This is a sample Reddit post text.. .",0
```

### Installation Steps

#### Step 1: Clone the Repository

```bash
git clone https://github.com/MyungwonKim/Language-and-AI-. git
cd Language-and-AI-
```

#### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install PyTorch with CUDA Support (for GPU training)

**For CUDA 11.8:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**Check your CUDA version:**
```bash
nvidia-smi
```

#### Step 4: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Download NLTK Data

```bash
python -c "import nltk; nltk. download('stopwords'); nltk.download('punkt')"
```

#### Step 6: Verify Installation

```python
import pandas as pd
import torch
import transformers

print(f"Pandas:  {pd.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
```

### Running the Full Pipeline

**Complete reproduction (all three models):**

```bash
# 1. Preprocess raw data (removes bots, non-English, fixes encoding)
python preprocessing_lang_ai.py

# 2. Filter posts containing self-descriptions (1st pass)
python regex_exp.py

# 3. Chunk long posts into manageable segments
python chunking.py

# 4. Filter chunks to keep only those with self-descriptions (2nd pass)
python regex_exp.py  # Run again on chunked data

# 5. Create three dataset variants
# 5a. Raw dataset:  use output from step 4 as-is
# 5b.  Masked self-descriptions
python masking. py
# 5c. Random masking (control)
python masking_random. py

# 6. Split into train/val/test (70/15/15 by author)
# Edit file paths in traintest_split.py for each dataset
python traintest_split. py  # Run 3 times (raw, masked_self, masked_random)

# 7. Train RoBERTa models
# Edit TRAIN_FILE, VAL_FILE, OUTPUT_DIR in RobertaBase.py for each run
python RobertaBase. py  # Run 3 times (one per dataset)

# 8. Compare model performances
python metrics_eval.py

# 9. Analyze word importance
python word_importance.py
```


### Expected Outputs

After running the full pipeline, you should have:

1. **Processed datasets:**
   - `data/processed_reddit_posts.csv` (cleaned data)
   - `data/MBTI_self_descriptions_only.csv` (filtered data - 1st pass)
   - `data/chunked_data.csv` (chunked posts)
   - `data/MBTI_self_descriptions_chunked.csv` (filtered chunks - 2nd pass)
   - `data/dataset_masked_v2.csv` (masked self-descriptions)
   - `data/dataset_random_masked.csv` (random masking)

2. **Train/val/test splits** in `data/raw_data/`, `data/masked_self/`, `data/masked_random/`

3. **Trained models** in `data/roberta_output_*/final_model/`

4. **Evaluation outputs:**
   - Training curves:  `loss_curve.png`
   - Confusion matrices: `confusion_matrix.png`
   - F1 threshold analysis: `f1_threshold_curve.png`
   - Model comparison: `model_comparison_with_baseline.png`
   - Metrics CSV: `model_comparison_results.csv`
   - Model feature importance: 'global_word_importance.png'

## ⚙️ Experimental Configuration

This section describes key parameters you can modify to change the experiments.  

### Main Training Hyperparameters

**File:** `RobertaBase.py`

| Parameter | Line | Default Value | Description | Recommended Range |
|-----------|------|---------------|-------------|-------------------|
| `MODEL_NAME` | 22 | `"roberta-base"` | Base transformer model | `"roberta-base"`, `"roberta-large"`, `"distilroberta-base"` |
| `MAX_LEN` | 23 | `512` | Maximum sequence length (RoBERTa limit) | 128-512 |
| `CHUNK_SIZE` | 24 | `350` | Target tokens per chunk | 200-500 |
| `BATCH_SIZE` | 25 | `4` | Training batch size | 2-16 (depends on GPU memory) |
| `EPOCHS` | 26 | `5` | Number of training epochs | 3-10 |
| `LEARNING_RATE` | 27 | `1e-5` | AdamW optimizer learning rate | 5e-6 to 5e-5 |


```

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Usage

### Step 1: Data Preprocessing

Start with the `extrovert_introvert.csv` file provided by the course.  

```bash
python preprocessing_lang_ai.py
```

This script cleans the raw data and prepares it for further processing.

### Step 2: Filter Self-Description Posts (1st Pass)

```bash
python regex_exp.py
```

This filters posts that include self-descriptions like "I'm an introvert" or "I'm an extrovert".  The output serves as the initial filtered dataset.

### Step 3: Chunk Long Posts

```bash
python chunking.py
```

This slices long posts into manageable chunks, ensuring they don't exceed the model's maximum token limit (512 tokens for RoBERTa).

### Step 4: Filter Chunks with Self-Descriptions (2nd Pass)

```bash
python regex_exp.py
```

Run the regex filter again on the chunked data to keep only chunks that contain self-descriptions.  This ensures that after chunking, we still have self-description information in each chunk.


### Step 5: Apply Masking Strategies

Generate three different datasets:

**Dataset 1: Raw data** (no masking)
- Use the output from Step 4 directly

**Dataset 2: Masked self-descriptions**
```bash
python masking.py
```
This masks all self-description phrases in the posts.

**Dataset 3: Random masking**
```bash
python masking_random. py
```
This randomly masks words in the posts for baseline comparison.

### Step 6: Split Data

For each of the three datasets, run: 

```bash
python traintest_split.py
```

This splits each dataset into train, test, and validation CSV files.

**Important:** After splitting, move each set of files (train.csv, test.csv, val.csv) into separate folders to avoid overwriting:  
- `data/raw_data/`
- `data/masked_self/`
- `data/masked_random/`

### Step 7: Train RoBERTa Models

Train three separate models, one for each dataset:

```bash
python RobertaBase.py
```

**Note:** You need to run this script three times, changing the data path each time to point to:  
1. Raw data folder
2. Masked self-description data folder
3. Random masked data folder

The trained models will be saved for later evaluation.  

### Step 8: Evaluate and Compare

Run the evaluation script to compare model performances:

```bash
python metrics_eval.py
```

This generates performance metrics (accuracy, F1-score, precision, recall) for all three models.

### Step 9: Analyze Word Importance

Understand which words the model relies on for predictions:

```bash
python word_importance.py
```

This generates bar charts showing the most important words for predicting introvert vs extrovert classes.

## File Descriptions

| File | Description |
|------|-------------|
| `preprocessing_lang_ai.py` | Cleans and preprocesses the raw CSV data |
| `regex_exp.py` | Filters posts/chunks containing self-description patterns (run twice) |
| `chunking.py` | Slices long posts into manageable chunks |
| `masking.py` | Masks self-description phrases in posts |
| `masking_random.py` | Randomly masks words in posts |
| `traintest_split.py` | Splits data into train/test/validation sets |
| `RobertaBase.py` | Trains RoBERTa model on the prepared datasets |
| `metrics_eval.py` | Evaluates and compares model performances |
| `word_importance.py` | Analyzes and plots word importance for each class |
| `baseline.py` | Baseline model implementation (TF-IDF + LogReg) |
| `performance_comparison.py` | Additional performance comparison utilities |
| `eda. ipynb` | Exploratory data analysis notebook |
| `requirements. txt` | Core dependencies for the pipeline |

## Project Structure

```
Language-and-AI-/
├── preprocessing_lang_ai.py
├── regex_exp.py
├── chunking.py
├── masking.py
├── masking_random.py
├── traintest_split. py
├── RobertaBase.py
├── metrics_eval.py
├── word_importance.py
├── baseline. py
├── eda.ipynb
├── requirements.txt
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
    ├── masked_random/
    │   ├── train.csv
    │   ├── test.csv
    │   └── val.csv
    └── roberta_output_*/
        ├── final_model/
        ├── loss_curve.png
        ├── confusion_matrix.png
        └── f1_threshold_curve.png
```

## Expected Results

The project aims to determine which masking strategy produces the best personality prediction model. Expected insights include:  
- Whether removing explicit self-descriptions improves model generalization
- How random masking affects model performance
- The importance of self-referential language in personality prediction


## Authors

- Group 29 Members

---
