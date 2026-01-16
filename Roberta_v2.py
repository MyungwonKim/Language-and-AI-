import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import os

# Configuration
TRAIN_FILE = "path/to/train.csv" # e.g., "assignment_data/masked_random/train.csv"
VAL_FILE   = "path/to/val.csv" # e.g., "assignment_data/masked_random/val.csv"
TEST_FILE  = "path/to/test.csv" # e.g., "assignment_data/masked_random/test.csv"

OUTPUT_DIR = "path/to/output_dir" # e.g., "assignment_data/roberta_output_masked_random"

# Model and Training Hyperparameters
MODEL_NAME = "roberta-base"
MAX_LEN = 512 
BATCH_SIZE = 4    
EPOCHS = 5
LEARNING_RATE = 1e-5

TEXT_COL = "final_text_random_masked"      
LABEL_COL = "extrovert" 

# Plotting style
plt.rcParams.update({'font.size': 12})
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12


# Batch tokenization function
def batch_tokenize(texts, tokenizer, batch_size=1000, desc="Tokenizing"):
    input_ids = []
    attention_masks = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        batch = [str(t) for t in batch]
        
        encodings = tokenizer(
            batch, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LEN,
            return_tensors=None 
        )
        input_ids.extend(encodings['input_ids'])
        attention_masks.extend(encodings['attention_mask'])
        
    return {'input_ids': input_ids, 'attention_mask': attention_masks}

# Custom Dataset Class
class PersonalityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        if 'input_ids' in encodings:
            self.input_ids = encodings['input_ids']

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def plot_evaluation(trainer, dataset, output_dir, dataset_name="Test"):
    """
    Generates a Normalized Confusion Matrix and an F1 vs Threshold Curve on the specified dataset.
    """
    print(f"\nGenerating plots ({dataset_name} Set)")
    predictions = trainer.predict(dataset)
    logits = torch.tensor(predictions.predictions)
    true_labels = predictions.label_ids
    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].numpy()
    
    # Normalized Confusion Matrix
    pred_labels = (probs >= 0.5).astype(int)
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Introvert', 'Extrovert'], 
                yticklabels=['Introvert', 'Extrovert'],
                annot_kws={"size": 18, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontsize=16, fontweight='bold', labelpad=10)
    plt.title(f'Normalized Confusion Matrix ({dataset_name})', fontsize=20, pad=20, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Normalized Confusion Matrix saved to {cm_path}")

    # F1 vs Threshold Curve
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_scores = [f1_score(true_labels, (probs >= t).astype(int), average='binary') for t in thresholds]
        
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='#ff7f0e', linewidth=4)
    plt.axvline(best_thresh, color='#d62728', linestyle='--', linewidth=2.5, label=f'Best Threshold: {best_thresh:.2f}')
    plt.scatter(best_thresh, best_f1, color='#d62728', s=150, zorder=5, edgecolor='black')
    
    plt.title(f'F1 Score vs. Decision Threshold ({dataset_name})', fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('Probability Threshold (Extrovert)', fontsize=16, labelpad=10)
    plt.ylabel('F1 Score', fontsize=16, labelpad=10)
    plt.legend(fontsize=14, loc='lower center', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    curve_path = os.path.join(output_dir, "f1_threshold_curve.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"F1 Curve saved to {curve_path}")

def plot_loss_history(trainer, output_dir):
    """
    Generates a Training vs Validation Loss Curve.
    """
    history = trainer.state.log_history
    train_loss, train_steps, val_loss, val_steps = [], [], [], []
    
    for entry in history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            train_steps.append(entry['step'])
        elif 'eval_loss' in entry:
            val_loss.append(entry['eval_loss'])
            val_steps.append(entry['step'])
            
    plt.figure(figsize=(12, 8))
    plt.plot(train_steps, train_loss, label='Training Loss', color='#1f77b4', linewidth=4, alpha=0.9)
    plt.plot(val_steps, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=4)
    
    plt.title('Learning Curve', fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=16, labelpad=10)
    plt.ylabel('Loss', fontsize=16, labelpad=10)
    plt.legend(fontsize=14, frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Loss Curve saved to {save_path}")


# Main execution
def main():
    # Load Data
    print("Loading data...")
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(VAL_FILE):
        print("Error: Train or Val file not found.")
        return
    
    # Check if test file exists
    has_test_set = False
    if os.path.exists(TEST_FILE):
        print("Test file found.")
        has_test_set = True
    else:
        print(f"Warning: Test file not found at {TEST_FILE}. Plotting will default to Validation set.")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    
    # Sanity Check (Drop NaNs)
    print("Cleaning data...")
    train_df = train_df.dropna(subset=[TEXT_COL])
    val_df = val_df.dropna(subset=[TEXT_COL])
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Training Samples:   {len(train_df)}")
    print(f"Validation Samples: {len(val_df)}")
    
    # Tokenize
    print("\nTokenizing")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    train_texts = train_df[TEXT_COL].astype(str).tolist()
    val_texts = val_df[TEXT_COL].astype(str).tolist()
    
    train_encodings = batch_tokenize(train_texts, tokenizer, desc="Tokenizing Train")
    val_encodings = batch_tokenize(val_texts, tokenizer, desc="Tokenizing Val  ")
    
    # Create Datasets
    train_dataset = PersonalityDataset(train_encodings, train_df[LABEL_COL].tolist())
    val_dataset = PersonalityDataset(val_encodings, val_df[LABEL_COL].tolist())
    
    # Initialize Model
    print("\nInitializing model")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=16,
        warmup_steps=10,
        weight_decay=0.1,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch", 
        save_strategy="epoch",       
        load_best_model_at_end=True, 
        metric_for_best_model="f1", 
        fp16=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    
    # Train
    print("\nTraining start")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Evaluation & Plots
    print("\nFinal evaluation")
    
    # Save Model
    model_save_path = OUTPUT_DIR + "/final_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot Loss (Uses Training and Validation History)
    plot_loss_history(trainer, OUTPUT_DIR)
    
    # Plotting on Test Set
    if has_test_set:
        print("\nPreparing Test Set for Plots")
        test_df = pd.read_csv(TEST_FILE)
        test_df = test_df.dropna(subset=[TEXT_COL])
        
        print(f"Test Samples: {len(test_df)}")
        
        test_texts = test_df[TEXT_COL].astype(str).tolist()
        test_encodings = batch_tokenize(test_texts, tokenizer, desc="Tokenizing Test")
        test_dataset = PersonalityDataset(test_encodings, test_df[LABEL_COL].tolist())
        test_results = trainer.predict(test_dataset)
        print("Test Set Metrics:", compute_metrics(test_results))
        
        # Plot using test dataset
        plot_evaluation(trainer, test_dataset, OUTPUT_DIR, dataset_name="Test")
    else:
        # Fallback to Validation if test set is missing
        print("Plotting Validation set (Test set missing)...")
        plot_evaluation(trainer, val_dataset, OUTPUT_DIR, dataset_name="Validation")

if __name__ == "__main__":
    main()