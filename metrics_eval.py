import pandas as pd
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import transformers

# Suppress huggingface warnings
transformers.logging.set_verbosity_error()

# Define model paths and the specific text column used for each
model_config = {
    "Model Raw Data": {
        "model_path": "Language-and-AI-/roberta_output_raw_chunk/final_model",
        "test_file": "assignment_data/raw_data/test.csv",
        "text_col": "text"   
    },
    "Model Masked Self Desc.": {
        "model_path": "Language-and-AI-/roberta_output_masked_self_chunk/final_model",
        "test_file": "assignment_data/masked_self/test.csv",
        "text_col": "final_text_masked"   
    },
    "Model Masked Random": {
        "model_path": "Language-and-AI-/roberta_output_masked_random_chunk/final_model",
        "test_file": "assignment_data/masked_random/test.csv",
        "text_col": "final_text_random_masked" 
    }
}

MAX_LEN = 512
BATCH_SIZE = 8  
LABEL_COL = "extrovert"

# Helper functions
def batch_tokenize(texts, tokenizer):
    input_ids = []
    attention_masks = []
    
    for i in tqdm(range(0, len(texts), 1000), desc="Tokenizing batch"):
        batch = texts[i : i + 1000]
        batch = [str(t) for t in batch]
        
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
    
    # Drop rows with missing text in the target column
    if target_text_col not in df.columns:
        print(f"Error: Column '{target_text_col}' not found in CSV.")
        return None
        
    df = df.dropna(subset=[target_text_col])
    print(f"  Loaded {len(df)} rows from test file.")
    
    test_texts = df[target_text_col].astype(str).tolist()
    test_labels = df[LABEL_COL].tolist()
    
    test_encodings = batch_tokenize(test_texts, tokenizer)
    test_dataset = PersonalityDataset(test_encodings, test_labels)

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
        "Accuracy": 0.7625,
        "Precision": 0.8611,
        "Recall": 0.3360,
        "F1 Score": 0.4834
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
    plot_df = results_df.melt(id_vars="Model", 
                              value_vars=["Precision", "Recall", "F1 Score"], 
                              var_name="Metric", 
                              value_name="Score")
    
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--"})
    
    chart = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="viridis")
    
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=4, fontsize=12, fontweight='bold')
    
    plt.title('Performance Comparison: Baseline vs. RoBERTa Models', fontsize=20, pad=20, fontweight='bold')
    plt.ylabel('Score', fontsize=16, labelpad=10, fontweight='bold')
    plt.xlabel('Model Version', fontsize=16, labelpad=10, fontweight='bold')
    
    plt.xticks(fontsize=13, rotation=15)
    plt.yticks(np.arange(0, 1.2, 0.1), fontsize=13)
    plt.ylim(0, 1.15)
    
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
               fontsize=13, title_fontsize=14, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("model_comparison_with_baseline.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    print("Plot saved as model_comparison_with_baseline.png")

if __name__ == "__main__":
    main()
