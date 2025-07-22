"""
Evaluation and reporting utilities for IT ticket classification.
"""
import pandas as pd
import joblib
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


def summarize_predictions(df_test, y_col, pred_col):
    """Generate summary statistics for model predictions."""
    # Compare predicted and actual categories
    df_test = df_test.copy()
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

    print("Summary saved to 'summary.xlsx' and detailed results to 'output_with_correct.xlsx'.")
    return summary_df


def inspect_influential_words(model_path='naive_bayes_model.joblib', n=5):
    """Analyze the most influential words for each category."""
    model = joblib.load(model_path)

    # Get the vectorizer and classifier from the pipeline
    vectorizer_names = ['countvectorizer', 'tfidfvectorizer']
    vectorizer = None
    
    for name in vectorizer_names:
        if name in model.named_steps:
            vectorizer = model.named_steps[name]
            break
    
    if vectorizer is None:
        raise ValueError("Could not find vectorizer in model pipeline")
    
    nb = model.named_steps['multinomialnb']

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    print(f"Top {n} influential words for each category:")
    for i, category in enumerate(nb.classes_):
        # Get indices of the top n words with the highest log-probabilities for this category
        top_word_indices = nb.feature_log_prob_[i].argsort()[-n:][::-1]
        top_words = feature_names[top_word_indices]
        print(f"'{category}': {', '.join(top_words)}")


def cluster_texts(data_file="./data/all_tickets.xlsx", vectorizer=CountVectorizer):
    """Perform K-means clustering on ticket texts."""
    # Load your data
    df = pd.read_excel(data_file)
    num_unique_categories = df['category'].nunique()
    print(f"Number of unique categories: {num_unique_categories}")

    df['Combined_Text'] = df['title'].astype(str) + ' ' + df['body'].astype(str)

    # Vectorize the text
    vectorizer_instance = vectorizer(stop_words='english')
    X = vectorizer_instance.fit_transform(df['Combined_Text'])

    n_clusters = num_unique_categories
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    df.to_excel('cluster.xlsx', index=False)
    print(f"Clustering results saved to 'cluster.xlsx'")


def create_word_clouds(data_file="./data/all_tickets.xlsx", output_dir="./wordclouds"):
    """Generate word clouds for each ticket category."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_excel(data_file)
    df['Combined_Text'] = df['title'].astype(str) + ' ' + df['body'].astype(str)
    
    categories = df['category'].unique()
    
    for category in categories:
        category_text = ' '.join(df[df['category'] == category]['Combined_Text'])
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100
        ).generate(category_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {category}')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{category.replace("/", "_")}_wordcloud.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Word clouds saved to '{output_dir}' directory")
