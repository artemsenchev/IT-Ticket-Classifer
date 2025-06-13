import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import kagglehub
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os
from sklearn.model_selection import cross_val_score


def download_data(orig_File):
    # Download latest version
    path = kagglehub.dataset_download("aniketg11/supportticketsclassification")

    print("Path to dataset files:", path)

    # Find the CSV file in the downloaded directory
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break

    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Convert df to Excel for offline use
    df.to_excel(orig_File, index=False)


def split_train_test_data(original_file='original.xlsx', y_col='category'):
    # Load the Excel file    
    df = pd.read_excel(original_file)

    # Split the DataFrame into training and testing sets with stratification
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[y_col]  # replace 'Category' with your label column name
    )

    return train_df, test_df

def train_naive_bayes_classifier(train_df, y_col, combined_text_col, alpha=1.0, vectorizer=CountVectorizer):
    # Ensure the necessary columns are present
    # Combine 'title' and 'body' columns into a single text feature

    X = train_df[combined_text_col]
    y = train_df[y_col]

    # Train Naive Bayes model
    model = make_pipeline(vectorizer(stop_words='english'), MultinomialNB(alpha=alpha))
    model.fit(X, y)

    # Export the trained model
    joblib.dump(model, 'naive_bayes_model.joblib')

    return model

def run_Classifier(test_df, combined_text_col, pred_col):

    # Load trained model
    model = joblib.load('naive_bayes_model.joblib')

    # Get predictions and probabilities
    predicted_categories = model.predict(test_df[combined_text_col])
    predicted_probabilities = model.predict_proba(test_df[combined_text_col])

    # Add results to DataFrame
    test_df[pred_col] = predicted_categories
    test_df['Prediction_probability'] = np.max(predicted_probabilities, axis=1)    

    return test_df

def summarize_predictions(df_test, y_col, pred_col):
    # Load the test data with predictions

    # Compare predicted and actual categories
    df_test['Correct'] = df_test[pred_col] == df_test[y_col]

    # Calculate counts
    true_count = df_test['Correct'].sum()
    false_count = (~df_test['Correct']).sum()
    total = len(df_test)

    # Calculate percentages
    true_pct = (true_count / total) * 100
    false_pct = (false_count / total) * 100

    # Print summary
    print(f"Correct predictions (True): {true_count} ({true_pct:.2f}%)")
    print(f"Incorrect predictions (False): {false_count} ({false_pct:.2f}%)")

    # Save the dataframe with the new 'Correct' column
    df_test.to_excel('output_with_correct.xlsx', index=False)

    # Save summary to Excel
    summary_df = pd.DataFrame({
    'Result': ['True', 'False'],
    'Count': [true_count, false_count],
    'Percentage': [true_pct, false_pct]
    })
    summary_df.to_excel('summary.xlsx', index=False)

    print("Summary of prediction accuracy saved to 'summary.xlsx' and detailed results to 'output_with_correct.xlsx'.")

def inspect_influential_words():
    model = joblib.load('naive_bayes_model.joblib')

    vectorizer = model.named_steps['countvectorizer']
    nb = model.named_steps['multinomialnb']

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    n = 5 # Number of top words to display

    print(f"Top {n} influential words")
    for i, category in enumerate(nb.classes_):
        # Get indices of the top n words with the highest log-probabilities for this category
        top_word_indices = nb.feature_log_prob_[i].argsort()[-n:][::-1]
        top_words = feature_names[top_word_indices]
        print(f"'{category}': {', '.join(top_words)}")

def cluster_texts(vectorizer=CountVectorizer):

    file_loc = "./data/original.xlsx"
    # Load your data
    df = pd.read_excel(file_loc)
    num_unique_categories = df['category'].nunique()
    print(f"Number of unique categories: {num_unique_categories}")

    df['Combined_Text'] = df['title'].astype(str) + ' ' + df['body'].astype(str)

    # Vectorize the text
    vectorizer = vectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Combined_Text'])

    n_clusters = num_unique_categories
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    df.to_excel('cluster.xlsx', index=False)

def print_classification_report(results_df, y_col, pred_col):
    y_true = results_df[y_col]
    y_pred = results_df[pred_col]
    print(classification_report(y_true, y_pred))


def cross_validate_naive_bayes(train_df, y_col, combined_text_col, model, cv=5):
    X = train_df[combined_text_col]
    y = train_df[y_col]
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")

def grid_search(train_df, y_col, combined_text_col, vectorizer=CountVectorizer):
    
    vectorizer_name = vectorizer.__name__.replace("Vectorizer", "").lower()

    # Pipeline with text preprocessing and model
    pipeline = Pipeline([
        (vectorizer_name, vectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])

    # Define hyperparameter grid


    param_grid = {
        'clf__alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5, 10, 100],  # Tested values
        f'{vectorizer_name}__max_features': [1000, 5000, 10000]  # Tune vectorizer - rough coding here but it gets it done
    }

    # Initialize grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1  # Use all CPU cores
    )

    # Execute search
    grid_search.fit(train_df[combined_text_col], train_df[y_col])

    # Results
    print(f"Best alpha: {grid_search.best_params_['clf__alpha']}")
    print(f"Best accuracy: {grid_search.best_score_:.4f}")

    return pipeline, grid_search.best_params_['clf__alpha']

def main():
    
    os.makedirs('./data', exist_ok=True)
    orig_File = "./data/all_tickets.xlsx"
    y_col = 'category'
    pred_col='Predicted_category'
    combined_text_col = 'Combined_Text'

    if not os.path.exists(orig_File):
        download_data(orig_File=orig_File)

    vectorizer = [CountVectorizer, TfidfVectorizer]
    vectorizer = vectorizer[1]

    print("splitting data into training and testing sets...")
    train_df, test_df = split_train_test_data(original_file=orig_File, y_col=y_col)
    train_df[combined_text_col] = train_df['title'].astype(str) + ' ' + train_df['body'].astype(str)
    test_df[combined_text_col] = test_df['title'].astype(str) + ' ' + test_df['body'].astype(str)
   
    print("running grid search for hyperparameter tuning...")
    _, alpha = grid_search(train_df, y_col=y_col, combined_text_col=combined_text_col, vectorizer=vectorizer)

    print("training Naive Bayes classifier...")
    model = train_naive_bayes_classifier(train_df, y_col=y_col, combined_text_col=combined_text_col, alpha=alpha, vectorizer=vectorizer)    

    print("performing cross-validation...")
    cross_validate_naive_bayes(train_df, y_col, combined_text_col, model, cv=5)

    print("running classifier on test data...")
    #output_df = run_Classifier(test_df, combined_text_col=combined_text_col, pred_col=pred_col)

    #print("summarizing predictions...")
    #print_classification_report(output_df, y_col=y_col, pred_col='Predicted_category')
    #summarize_predictions(output_df, y_col=y_col, pred_col=pred_col)

    #inspect_influential_words()
    #cluster_texts()
    #create_word_clouds()

if __name__ == "__main__":
    main()