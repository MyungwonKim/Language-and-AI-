import pandas as pd
from transformers import RobertaTokenizer
import nltk
from tqdm import tqdm
import os
import re

# --- CONFIGURATION ---
INPUT_FILE = "data/MBTI_self_descriptions_only.csv"
OUTPUT_FILE = "./chunked_data.csv"

TEXT_COL = "clean_text"   # The column we need to split

# Model Settings
MODEL_NAME = "roberta-base"
MAX_TOKENS = 500  

# Chunking logic
def chunk_text(text, extra_data, tokenizer, max_tokens=500):
    """
    Splits text into chunks
    """
    if not isinstance(text, str): 
        return []
    
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        
        # Count tokens
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)
        
        # Check limits
        if current_token_count + sentence_token_count > max_tokens:
            # Save current chunk
            if current_chunk:
                joined_text = " ".join(current_chunk)
                if len(tokenizer.encode(joined_text, add_special_tokens=False)) > 10:
                    # Create the chunk object
                    chunk_item = {"text": joined_text}
                    chunk_item.update(extra_data) 
                    chunks.append(chunk_item)
            
            # Reset
            current_chunk = []
            current_token_count = 0
            
            # Handle Single Massive Sentence
            if sentence_token_count > max_tokens:
                chunk_item = {"text": sentence}
                chunk_item.update(extra_data)
                chunks.append(chunk_item)
            else:
                current_chunk.append(sentence)
                current_token_count = sentence_token_count
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
            
    # Save leftover chunk
    if current_chunk:
        joined_text = " ".join(current_chunk)
        if len(tokenizer.encode(joined_text, add_special_tokens=False)) > 10:
            chunk_item = {"text": joined_text}
            chunk_item.update(extra_data)
            chunks.append(chunk_item)
    
    return chunks

# Main Execution
if __name__ == "__main__":
    print(f"Loading Tokenizer: {MODEL_NAME}")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        exit()

    print(f"Reading file: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Check if text column exists
    if TEXT_COL not in df.columns:
        print(f"Error: Text column '{TEXT_COL}' not found.")
        print(f"Available: {list(df.columns)}")
        exit()
        
    all_chunks = []
    print("Starting Chunking Process (Preserving all columns)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking Rows"):
        # Extract the text
        text_content = row[TEXT_COL]
        
        # We drop the original large text column so we don't duplicate it in every chunk
        extra_data = row.drop(labels=[TEXT_COL]).to_dict()
        
        new_chunks = chunk_text(
            text_content, 
            extra_data, 
            tokenizer, 
            MAX_TOKENS
        )
        all_chunks.extend(new_chunks)
        
    # Convert to DataFrame
    chunked_df = pd.DataFrame(all_chunks)
    
    print(f"\nSummary")
    print(f"Original Rows: {len(df)}")
    print(f"Total Chunks:  {len(chunked_df)}")
    print(f"Columns Saved: {list(chunked_df.columns)}")
    
    # Save
    chunked_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to: {os.path.abspath(OUTPUT_FILE)}")
