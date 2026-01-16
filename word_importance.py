import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

MODEL_PATH = "path_to_saved_model_directory"  # Update this to your model path
TEST_FILE  = "path_to_test_data.csv"        # Update this to your test data path

save_path = "path_to_save_plots/word_importance_plot.png"  # Update this to your desired save path

TEXT_COL = "final_text_random_masked" # Chagnge this to the specific column you want to analyze for this run
SAMPLE_SIZE = 200


def analyze_global_importance(text_col_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device found: {device}")

    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.to(device)

    cls_explainer = SequenceClassificationExplainer(model, tokenizer)

    print(f"Loading Data from {TEST_FILE}...")
    df = pd.read_csv(TEST_FILE)

    # Validation Check
    if text_col_name not in df.columns:
        raise ValueError(f"Error: Column '{text_col_name}' not found. Available: {list(df.columns)}")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    texts = df[text_col_name].astype(str).tolist()

    word_scores = defaultdict(float)
    word_counts = defaultdict(int)

    print(f"\nStarting analysis on column '{text_col_name}' (Size: {len(texts)})")

    for text in tqdm(texts, desc="Analyzing Posts"):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=500,
            add_special_tokens=False
        )
        truncated_text = tokenizer.decode(encoded['input_ids'])

        try:
            attributions = cls_explainer(
                truncated_text,
                index=1,
                n_steps=10,
                internal_batch_size=2
            )

            for word, score in attributions:
                clean_word = word.replace('Ġ', '').lower().strip()

                stop_words = [
                    '<s>', '</s>', '.', ',', '!', '?', 'the', 'a', 'an', 'to', 'and',
                    'of', 'is', 'it', 'that', 'in', 'for', 'my', 'i', 'me', 'but',
                    'so', 'with', 'on', 'be', 'have', 'do', 'not', 'just', 'this',
                    'was', 'at', 'my', 'as'
                ]

                if clean_word in stop_words: continue
                if len(clean_word) < 3: continue

                word_scores[clean_word] += score
                word_counts[clean_word] += 1

        except Exception as e:
            print(f"Skipped a text due to error: {e}")
            continue

    avg_scores = {k: v / word_counts[k] for k, v in word_scores.items()}
    sorted_words = sorted(avg_scores.items(), key=lambda item: item[1], reverse=True)

    # Taking top 20 for each
    top_extrovert = sorted_words[:20]
    top_introvert = sorted_words[-20:][::-1]

    return top_extrovert, top_introvert

def plot_importance(top_extrovert, top_introvert):
    print("\nGenerating plots")
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 10)) 
    
    # Extrovert plot
    words_e, scores_e = zip(*top_extrovert)
    sns.barplot(x=list(scores_e), y=list(words_e), ax=axes[0], palette="Greens_r")
    
    # Matching style from plot_evaluation
    axes[0].set_title("Top Words: 'EXTROVERT'", fontsize=22, fontweight='bold', pad=20)
    axes[0].set_xlabel("Influence Score (Positive)", fontsize=18, fontweight='bold', labelpad=10)
    axes[0].set_ylabel("Words", fontsize=18, fontweight='bold', labelpad=10)
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    
    # Introvert plot
    words_i, scores_i = zip(*top_introvert)
    scores_i_abs = [abs(s) for s in scores_i] 
    sns.barplot(x=scores_i_abs, y=list(words_i), ax=axes[1], palette="Reds_r")
    
    # Matching style from plot_evaluation
    axes[1].set_title("Top Words: 'INTROVERT'", fontsize=22, fontweight='bold', pad=20)
    axes[1].set_xlabel("Influence Score (Negative Strength)", fontsize=18, fontweight='bold', labelpad=10)
    axes[1].set_ylabel("") # Redundant Y label
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout(pad=3.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    print(f"High-resolution plot saved to: {os.path.abspath(save_path)}")
    print("\nGenerating plots")


    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Extrovert Plot 
    words_e, scores_e = zip(*top_extrovert)
    sns.barplot(x=list(scores_e), y=list(words_e), ax=axes[0], palette="Greens_r")

    # Set titles
    axes[0].set_title("Top Words: 'EXTROVERT'", fontsize=20, fontweight='bold', pad=20)
    axes[0].set_xlabel("Influence Score (Positive)", fontsize=16, labelpad=15)
    axes[0].set_ylabel("Words", fontsize=16)

    axes[0].tick_params(axis='y', labelsize=14)


    # Introvert Plot
    words_i, scores_i = zip(*top_introvert)
    scores_i_abs = [abs(s) for s in scores_i]
    sns.barplot(x=scores_i_abs, y=list(words_i), ax=axes[1], palette="Reds_r")

    axes[1].set_title("Top Words: 'INTROVERT'", fontsize=20, fontweight='bold', pad=20)
    axes[1].set_xlabel("Influence Score (Negative Strength)", fontsize=16, labelpad=15)
    axes[1].set_ylabel("") # Remove redundant Y-label on second plot
    axes[1].tick_params(axis='y', labelsize=14)

    plt.tight_layout(pad=1.5, w_pad=3.0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    print(f"Pplot saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    extrovert_words, introvert_words = analyze_global_importance(TEXT_COL)

    print("\nTOP EXTROVERT WORDS")
    for w, s in extrovert_words:
        print(f"{w}: {s:.4f}")

    print("\nTOP INTROVERT WORDS")
    for w, s in introvert_words:
        print(f"{w}: {s:.4f}")

    plot_importance(extrovert_words, introvert_words)
