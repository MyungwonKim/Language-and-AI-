import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
file_path = "data/your_dataset.csv" # Change this to your dataset path
df = pd.read_csv(file_path)

# Unique authors
unique_authors = df['auhtor_ID'].unique()
print(f"Total unique authors: {len(unique_authors)}")

# Split authors (70% Train, 30% Temp)
train_authors, temp_authors = train_test_split(unique_authors, test_size=0.3, random_state=42)

# Split the 30% into half (15% Val, 15% Test)
val_authors, test_authors = train_test_split(temp_authors, test_size=0.5, random_state=42)

# Filter the dataframe
train_df = df[df['auhtor_ID'].isin(train_authors)]
val_df = df[df['auhtor_ID'].isin(val_authors)]
test_df = df[df['auhtor_ID'].isin(test_authors)]

# Check rows
print(f"Train Set: {len(train_df)} rows ({len(train_authors)} authors)")
print(f"Val Set:   {len(val_df)} rows ({len(val_authors)} authors)")
print(f"Test Set:  {len(test_df)} rows ({len(test_authors)} authors)")

# Save
train_df.to_csv("data/raw_data/train.csv", index=False) # Change file path depending on your dataset to split
val_df.to_csv("data/raw_data/val.csv", index=False)
test_df.to_csv("data/raw_data/test.csv", index=False)
