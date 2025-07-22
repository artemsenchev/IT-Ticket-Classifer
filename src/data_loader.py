"""
Data loading and preprocessing utilities for IT ticket classification.
"""
import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split


def download_data(orig_file):
    """Download the support tickets dataset from Kaggle."""
    # Download latest version
    path = kagglehub.dataset_download("aniketg11/supportticketsclassification")
    print("Path to dataset files:", path)

    # Find the CSV file in the downloaded directory
    csv_file = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break

    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Convert df to Excel for offline use
    df.to_excel(orig_file, index=False)


def split_train_test_data(original_file='original.xlsx', y_col='category', test_size=0.2):
    """Split data into training and testing sets with stratification."""
    # Load the Excel file    
    df = pd.read_excel(original_file)

    # Split the DataFrame into training and testing sets with stratification
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[y_col], random_state=42
    )

    return train_df, test_df


def preprocess_text_data(df, combined_text_col='Combined_Text'):
    """Combine title and body columns into a single text feature."""
    df[combined_text_col] = df['title'].astype(str) + ' ' + df['body'].astype(str)
    return df
