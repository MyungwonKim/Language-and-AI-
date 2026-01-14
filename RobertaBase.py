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
import nltk
import re

TRAIN_FILE = "data/raw_data/train.csv" # Change this to your training dataset path
VAL_FILE   = "data/raw_data/val.csv" # Change this to your validation dataset path
OUTPUT_DIR = "data/roberta_output_raw"  # Change this to your desired output directory

# Model and Training Hyperparameters
MODEL_NAME = "roberta-base"
MAX_LEN = 512 # Robertabase can handle up to 512 tokens
CHUNK_SIZE = 350
BATCH_SIZE = 4    
EPOCHS = 5
LEARNING_RATE = 1e-5

TEXT_COL = "clean_text" # Change if your text column has a different name
LABEL_COL = "extrovert"

def plot_evaluation(trainer, val_dataset, output_dir):
    """
    Generates a Confusion Matrix and an F1 vs Threshold Curve.
    """
    print("\n Generating Evaluation Plots")
    
    predictions = trainer.predict(val_dataset)
    logits = torch.tensor(predictions.predictions)
    true_labels = predictions.label_ids
    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].numpy()
    
    # Plot confusion matrix
    pred_labels = (probs >= 0.5).astype(int)
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Introvert', 'Extrovert'], 
                yticklabels=['Introvert', 'Extrovert'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Threshold = 0.5)')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")

    # Plot F1 Score vs Threshold Curve
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_scores = []
    
    for t in thresholds:
        preds_t = (probs >= t).astype(int)
        score = f1_score(true_labels, preds_t, average='binary')
        f1_scores.append(score)
        
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='orange', linewidth=2)
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold: {best_thresh:.2f}')
    plt.scatter(best_thresh, best_f1, color='red')
    
    plt.title('F1 Score vs. Decision Threshold')
    plt.xlabel('Probability Threshold for Class 1 (Extrovert)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    curve_path = os.path.join(output_dir, "f1_threshold_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"F1 Curve saved to {curve_path}")
    print(f"Optimal Threshold found: {best_thresh:.2f} (Max F1: {best_f1:.4f})")

def plot_loss_history(trainer, output_dir):
    # Extract logs
    history = trainer.state.log_history
    
    # Separate training and validation logs
    train_loss = []
    train_steps = []
    val_loss = []
    val_steps = []
    
    for entry in history:
        if 'loss' in entry:  
            train_loss.append(entry['loss'])
            train_steps.append(entry['step'])
        elif 'eval_loss' in entry:  
            val_loss.append(entry['eval_loss'])
            val_steps.append(entry['step'])
            
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(val_steps, val_loss, label='Validation Loss', color='orange', linewidth=2)
    
    plt.title('Learning Curve: Training vs. Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = output_dir + "/loss_curve.png"
    plt.savefig(save_path)
    print(f"Loss Curve saved to {save_path}")
    plt.show()

# Function for token based text chunking
def chunk_text(text, label, tokenizer, max_tokens=500):
    """
    Chunk text based on token count while respecting sentence boundaries.
    
    Parameters:
    - text: input text
    - label: class label
    - tokenizer: RobertaTokenizer instance
    - max_tokens: maximum tokens per chunk (512 is model limit, we use 500 for safety)
    
    Returns: list of dicts with 'text' and 'label'
    """
    if not isinstance(text, str): 
        return []
    
    # Split into sentences using nltk
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        # Split by sentence ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Count tokens in this sentence
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)
        
        # Check if adding this sentence exceeds max tokens
        if current_token_count + sentence_token_count > max_tokens:
            # Save current chunk if it has content
            if current_chunk:
                joined_text = " ".join(current_chunk)
                # Only keep chunks with meaningful content (> 20 tokens)
                chunk_token_count = len(tokenizer.encode(joined_text, add_special_tokens=False))
                if chunk_token_count > 20:
                    chunks.append({
                        "text": joined_text,
                        "label": int(label)
                    })
            
            # Reset for new chunk
            current_chunk = []
            current_token_count = 0
            
            # if single sentence is > max_tokens, still add it (the tokenizer will truncate it at 512, but we preserve as much as possible)
            if sentence_token_count > max_tokens:
                chunks.append({
                    "text": sentence,
                    "label": int(label)
                })
            else:
                # Start new chunk with this sentence
                current_chunk.append(sentence)
                current_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
    
    # Add the final leftover chunk
    if current_chunk:
        joined_text = " ".join(current_chunk)
        chunk_token_count = len(tokenizer.encode(joined_text, add_special_tokens=False))
        if chunk_token_count > 20:
            chunks.append({
                "text": joined_text,
                "label": int(label)
            })
    
    return chunks

def prepare_dataset(df, tokenizer, desc="Chunking Data"):
    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        text = row[TEXT_COL]
        label = row[LABEL_COL]
        new_rows.extend(chunk_text(text, label, tokenizer))
    return pd.DataFrame(new_rows)

# Batch tokenization function
def batch_tokenize(texts, tokenizer, batch_size=1000, desc="Tokenizing"):
    input_ids = []
    attention_masks = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
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

# Main execution
def main():
    # Load data
    print("Loading training and validation data")
    train_raw = pd.read_csv(TRAIN_FILE)
    val_raw = pd.read_csv(VAL_FILE)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Chunk Data 
    print("\nChunking text")
    train_chunked = prepare_dataset(train_raw, tokenizer, desc="Processing Train Set")
    val_chunked = prepare_dataset(val_raw, tokenizer, desc="Processing Val Set  ")
    
    print(f"Original Train Chunks: {len(train_chunked)}")
    print(f"Original Val Chunks:   {len(val_chunked)}")
    
    train_final = train_chunked.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Training on dataset size: {len(train_final)}")
    print(f"Class Distribution:\n{train_final['label'].value_counts()}")
    
    print("\nTokenizing")
    
    train_texts = train_final['text'].astype(str).tolist()
    val_texts = val_chunked['text'].astype(str).tolist()
    
    train_encodings = batch_tokenize(train_texts, tokenizer, desc="Tokenizing Train")
    val_encodings = batch_tokenize(val_texts, tokenizer, desc="Tokenizing Val  ")
    
    # Create datasets
    train_dataset = PersonalityDataset(train_encodings, train_final['label'].tolist())
    val_dataset = PersonalityDataset(val_encodings, val_chunked['label'].tolist())
    
    # Initialize model
    print("\nInitializing model")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=16,
        warmup_steps=60,
        weight_decay=0.1,
        logging_dir='./logs',
        logging_steps=30,
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
    
    # Final Evaluation
    print("\nFinal evaluation")
    results = trainer.evaluate()
    print("Results:", results)
    
    # Save model
    model_save_path = OUTPUT_DIR + "/final_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # Plot visualizations
    plot_loss_history(trainer, OUTPUT_DIR)
    plot_evaluation(trainer, val_dataset, OUTPUT_DIR)

if __name__ == "__main__":
    main()
