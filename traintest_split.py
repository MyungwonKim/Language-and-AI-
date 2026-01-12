import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Use the file that has the FULL text (no truncation)
INPUT_FILE = r"C:/Users/20235050/Downloads/BDS_Y3/Language_AI/assignment_data/processed_reddit_authors.csv"
OUTPUT_DIR = r"C:/Users/20235050/Downloads/BDS_Y3/Language_AI/assignment_data/"

# Column Names
LABEL_COL = "extrovert"  # The column to stratify by (0 or 1)

# Split Ratios
TEST_SIZE = 0.15  # 15% for Testing
VAL_SIZE = 0.15   # 15% for Validation
# Remaining 70% will be Training

def split_dataset():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='latin1')

    # Drop any broken rows (missing labels)
    df = df.dropna(subset=[LABEL_COL])
    
    print(f"Total Authors: {len(df)}")
    
    # --- STEP 1: Split off the TEST set ---
    # We use 'stratify' to keep the Introvert/Extrovert ratio consistent
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        random_state=42, 
        stratify=df[LABEL_COL]
    )
    
    # --- STEP 2: Split the remaining into TRAIN and VALIDATION ---
    # Adjust val_size to be relative to the remaining data
    # If we want 15% total for Val, and we have 85% left, 
    # we need (0.15 / 0.85) = ~17.6% of the remaining chunk.
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_size, 
        random_state=42, 
        stratify=train_val_df[LABEL_COL]
    )
    
    # --- REPORTING ---
    print("-" * 30)
    print(f"Training Set:   {len(train_df)} authors ({len(train_df)/len(df):.1%})")
    print(f"Validation Set: {len(val_df)} authors ({len(val_df)/len(df):.1%})")
    print(f"Testing Set:    {len(test_df)} authors ({len(test_df)/len(df):.1%})")
    print("-" * 30)
    
    # Verify Class Balance
    print("\nClass Balance (Percentage of Extroverts):")
    print(f"Original: {df[LABEL_COL].mean():.1%}")
    print(f"Train:    {train_df[LABEL_COL].mean():.1%}")
    print(f"Val:      {val_df[LABEL_COL].mean():.1%}")
    print(f"Test:     {test_df[LABEL_COL].mean():.1%}")
    
    # --- SAVING ---
    print("\nSaving files...")
    train_df.to_csv(OUTPUT_DIR + "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR + "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR + "test.csv", index=False)
    print("Done! Files saved as train.csv, val.csv, and test.csv")

if __name__ == "__main__":
    split_dataset()