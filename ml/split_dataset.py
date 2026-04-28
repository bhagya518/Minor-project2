"""
Split dataset into train and test sets
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("../dataset (1).csv")

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_malicious'])

# Save splits
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Dataset split complete:")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Saved to train.csv and test.csv")
