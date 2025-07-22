"""
Configuration settings for the IT ticket classifier.
"""

# Data settings
DATA_DIR = "./data"
ORIGINAL_FILE = "./data/all_tickets.xlsx"
MODEL_FILE = "naive_bayes_model.joblib"

# Column names
CATEGORY_COLUMN = 'category'
PREDICTION_COLUMN = 'Predicted_category'
COMBINED_TEXT_COLUMN = 'Combined_Text'
TITLE_COLUMN = 'title'
BODY_COLUMN = 'body'

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Grid search parameters
ALPHA_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5, 10, 100]
MAX_FEATURES_VALUES = [1000, 5000, 10000]

# Output files
OUTPUT_WITH_CORRECT_FILE = 'output_with_correct.xlsx'
SUMMARY_FILE = 'summary.xlsx'
CLUSTER_FILE = 'cluster.xlsx'
WORDCLOUD_DIR = './wordclouds'
