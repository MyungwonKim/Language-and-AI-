import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

FILE_PATH = "/assignment_data/extrovert_introvert.csv" # Update to your raw file path
TEXT_COL = "post" 
TOP_N = 20        # Number of top words to show

def main():
    # Load Data
    print(f"Loading data from {FILE_PATH}...")
    if not os.path.exists(FILE_PATH):
        print("Error: File not found.")
        return
    
    df = pd.read_csv(FILE_PATH)
    
    if TEXT_COL not in df.columns:
        print(f"Error: Column '{TEXT_COL}' not found. Available: {list(df.columns)}")
        return

    # Drop empty rows
    df = df.dropna(subset=[TEXT_COL])
    
    # Tokenize (split by whitespace)
    print("Counting words...")
    all_text = " ".join(df[TEXT_COL].astype(str).tolist())
    words = all_text.split()
    
    # Analyze Vocab
    word_counts = Counter(words)
    vocab_size = len(word_counts)
    total_tokens = len(words)
    
    print(f"\nVocabulary Statistics")
    print(f"Total Words (Tokens): {total_tokens:,}")
    print(f"Vocabulary Size (Unique Words): {vocab_size:,}")
    
    # Get Top Words
    most_common = word_counts.most_common(TOP_N)
    words_x, counts_y = zip(*most_common)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(x=list(counts_y), y=list(words_x), palette="viridis")
    
    plt.title(f'Top {TOP_N} Most Frequent Words (Raw Data)', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Frequency', fontsize=14, fontweight='bold')
    plt.ylabel('Word', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add counts at the end of bars
    for i, v in enumerate(counts_y):
        plt.text(v + (max(counts_y)*0.01), i, f"{v:,}", va='center', fontsize=10)
        
    plt.tight_layout()
    save_path = "raw_vocab_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
