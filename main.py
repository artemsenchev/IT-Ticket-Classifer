"""
Main script for running the IT ticket classifier.
"""
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.data_loader import download_data, split_train_test_data, preprocess_text_data
from src.model import TicketClassifier, cross_validate_model, grid_search_hyperparameters, print_classification_report
from src.evaluation import summarize_predictions, inspect_influential_words, cluster_texts, create_word_clouds
from src.config import *


def main():
    """Main execution function."""
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download data if not exists
    if not os.path.exists(ORIGINAL_FILE):
        print("Downloading dataset...")
        download_data(ORIGINAL_FILE)

    # Choose vectorizer
    vectorizer = TfidfVectorizer  # or CountVectorizer

    print("Splitting data into training and testing sets...")
    train_df, test_df = split_train_test_data(
        original_file=ORIGINAL_FILE, 
        y_col=CATEGORY_COLUMN,
        test_size=TEST_SIZE
    )
    
    # Preprocess text data
    train_df = preprocess_text_data(train_df, COMBINED_TEXT_COLUMN)
    test_df = preprocess_text_data(test_df, COMBINED_TEXT_COLUMN)

    print("Running grid search for hyperparameter tuning...")
    best_params = grid_search_hyperparameters(
        train_df, 
        y_col=CATEGORY_COLUMN, 
        combined_text_col=COMBINED_TEXT_COLUMN, 
        vectorizer=vectorizer
    )

    print("Training optimized Naive Bayes classifier...")
    classifier = TicketClassifier(
        vectorizer=vectorizer, 
        alpha=best_params['clf__alpha']
    )
    model = classifier.train(train_df, CATEGORY_COLUMN, COMBINED_TEXT_COLUMN)
    classifier.save_model(MODEL_FILE)

    print("Performing cross-validation...")
    cross_validate_model(
        train_df, 
        CATEGORY_COLUMN, 
        COMBINED_TEXT_COLUMN, 
        model, 
        cv=CV_FOLDS
    )

    print("Running classifier on test data...")
    output_df = classifier.predict(
        test_df, 
        combined_text_col=COMBINED_TEXT_COLUMN, 
        pred_col=PREDICTION_COLUMN
    )

    print("Generating evaluation reports...")
    print_classification_report(output_df, y_col=CATEGORY_COLUMN, pred_col=PREDICTION_COLUMN)
    summarize_predictions(output_df, y_col=CATEGORY_COLUMN, pred_col=PREDICTION_COLUMN)

    # Optional analysis
    #print("\nRunning additional analysis...")
    #inspect_influential_words(MODEL_FILE)
    
    # Uncomment these if you want to run clustering and word cloud generation
    # cluster_texts(ORIGINAL_FILE)
    # create_word_clouds(ORIGINAL_FILE, WORDCLOUD_DIR)


if __name__ == "__main__":
    main()
