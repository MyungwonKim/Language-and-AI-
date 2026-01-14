import pandas as pd
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import tqdm
import os
import transformers
import re

# Suppress huggingface warnings
transformers.logging.set_verbosity_error()

# Define model paths and the specific text column used for each
# Change paths and column names as needed
model_config = {
    "Model Raw Data": {
        "model_path": "./roberta_output_raw/final_model",
        "test_file": "data/raw_data/test.csv",
        "text_col": "clean_text"   
    },
    "Model Masked Self Desc.": {
        "model_path": "./roberta_output_masked_self/final_model",
        "test_file": "data/masked_self/test.csv",
        "text_col": "final_text_masked"   
    },
    "Model Masked Random": {
        "model_path": "./roberta_output_masked_random/final_model",
        "test_file": "data/masked_random/test.csv",
        "text_col": "final_text_random_masked" 
    }
}

MAX_LEN = 512
BATCH_SIZE = 16
LABEL_COL = "extrovert"

# Chunking logic
def chunk_text(text, label, tokenizer, max_tokens=500):
    """
    Chunks text based on token count while respecting sentence boundaries.
    """
    if not isinstance(text, str): return []
    
    # Attempt to split by sentences first
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)
        
        # Check if adding this sentence exceeds the limit
        if current_token_count + sentence_token_count > max_tokens:
            if current_chunk:
                joined_text = " ".join(current_chunk)
                # Filter out tiny chunks that are likely noise
                if len(tokenizer.encode(joined_text, add_special_tokens=False)) > 20:
                    chunks.append({"text": joined_text, "label": int(label)})
            
            current_chunk = []
            current_token_count = 0
            
            # If a single sentence is too long, we have to add it anyway (will get truncated later)
            if sentence_token_count > max_tokens:
                chunks.append({"text": sentence, "label": int(label)})
            else:
                current_chunk.append(sentence)
                current_token_count = sentence_token_count
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
    
    # Add any remaining text
    if current_chunk:
        joined_text = " ".join(current_chunk)
        if len(tokenizer.encode(joined_text, add_special_tokens=False)) > 20:
            chunks.append({"text": joined_text, "label": int(label)})
    
    return chunks

# Dataset preparation
def prepare_dataset(df, tokenizer, text_col_name):
    new_rows = []
    
    if text_col_name not in df.columns:
        raise ValueError(f"Error: Column '{text_col_name}' not found. Available: {list(df.columns)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Chunking ({text_col_name})"):
        text = row[text_col_name]
        label = row[LABEL_COL]
        new_rows.extend(chunk_text(text, label, tokenizer))
        
    return pd.DataFrame(new_rows)

# Helper functions
def batch_tokenize(texts, tokenizer):
    input_ids = []
    attention_masks = []
    
    for i in tqdm(range(0, len(texts), 1000), desc="Tokenizing batch"):
        batch = texts[i : i + 1000]
        encodings = tokenizer(
            batch, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LEN
        )
        input_ids.extend(encodings['input_ids'])
        attention_masks.extend(encodings['attention_mask'])
        
    return {'input_ids': input_ids, 'attention_mask': attention_masks}

class PersonalityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
        
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Evaluation pipeline
def evaluate_model(name, config):
    model_path = config["model_path"]
    test_file_path = config["test_file"]
    target_text_col = config["text_col"]
    
    print(f"\nProcessing: {name}...")
    print(f"Model path: {model_path}")
    print(f"Test data: {test_file_path}")
    print(f"Target column: {target_text_col}")
    
    # Load tokenizer
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
    except Exception:
        print(f"  Warning: Local tokenizer not found. Using 'roberta-base'.")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load model
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error: Could not load model for {name}. {e}")
        return None

    if not os.path.exists(test_file_path):
        print(f"Error: File not found at {test_file_path}")
        return None
        
    df = pd.read_csv(test_file_path)
    
    # Prepare data with specific column
    try:
        chunked_df = prepare_dataset(df, tokenizer, target_text_col)
    except ValueError as e:
        print(e)
        return None
    
    if chunked_df.empty:
        return None
    
    print(f"  Generated {len(chunked_df)} chunks.")
    
    test_texts = chunked_df['text'].astype(str).tolist()
    test_encodings = batch_tokenize(test_texts, tokenizer)
    test_dataset = PersonalityDataset(test_encodings, chunked_df['label'].tolist())

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./eval_temp", per_device_eval_batch_size=BATCH_SIZE, report_to="none"),
        compute_metrics=compute_metrics
    )
    
    predictions = trainer.predict(test_dataset)
    metrics = predictions.metrics
    
    return {
        "Model": name,
        "Accuracy": metrics['test_accuracy'],
        "Precision": metrics['test_precision'],
        "Recall": metrics['test_recall'],
        "F1 Score": metrics['test_f1']
    }

def main():
    results = []
    
    # Add manual baseline results for comparison
    baseline_result = {
        "Model": "Baseline (TF-IDF)",
        "Accuracy": 0.7630,
        "Precision": 0.8901,
        "Recall": 0.2883,
        "F1 Score": 0.4355
    }
    results.append(baseline_result)

    # Evaluate models
    for model_name, config in model_config.items():
        res = evaluate_model(model_name, config)
        if res: results.append(res)

    if not results: return

    # Process and visualize results
    results_df = pd.DataFrame(results)
    print("\nFinal comparison table")
    print(results_df)

    results_df.to_csv("model_comparison_results.csv", index=False)

    # Plotting
    plot_df = results_df.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1 Score"], var_name="Metric", value_name="Score")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    
    chart = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="viridis")
    
    # Add numerical labels on bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=3)
    
    plt.title('Performance comparison', fontsize=16)
    plt.ylim(0, 1.15)
    plt.ylabel('Score')
    plt.xlabel('Model version')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    plt.savefig("model_comparison_with_baseline.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
