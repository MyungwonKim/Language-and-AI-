import pandas as pd
import numpy as np
import re

# load data
d_mask = pd.read_csv(r"C:\Users\jelic\Desktop\L&AI\this.csv")

PLACEHOLDER = "[MBTI_SELF_DESC]"
SEED = 42
rng = np.random.default_rng(SEED)

def count_words(s):
    if not isinstance(s, str):
        return 0
    return len(re.findall(r"\b\w+\b", s))

def random_consecutive_word_mask(text, n_words, rng):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    if n_words <= 0:
        return text

    # tokenize: words | spaces | punctuation
    tokens = re.findall(r"\b\w+\b|\s+|[^\w\s]+", text)
    word_positions = [i for i, t in enumerate(tokens) if re.fullmatch(r"\b\w+\b", t)]

    if not word_positions:
        return text

    n_words = min(n_words, len(word_positions))
    start = rng.integers(0, len(word_positions) - n_words + 1)
    span = word_positions[start:start + n_words]

    tokens[span[0]] = PLACEHOLDER
    for pos in span[1:]:
        tokens[pos] = ""

    return "".join(tokens)

# number of words to mask = number of words in matched_text
d_mask["mask_word_count"] = d_mask["matched_text"].apply(count_words)

# apply random masking
d_mask["final_text_random_masked"] = d_mask.apply(
    lambda row: random_consecutive_word_mask(
        row["clean_text"],
        row["mask_word_count"],
        rng
    ),
    axis=1
)

# save
d_mask.to_csv("dataset_random_masked.csv", index=False)

print(d_mask["final_text_random_masked"].head(5))
