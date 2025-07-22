"""
Model training, validation, and hyperparameter tuning for IT ticket classification.
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report


class TicketClassifier:
    """Naive Bayes classifier for IT support tickets."""
    
    def __init__(self, vectorizer=TfidfVectorizer, alpha=1.0):
        self.vectorizer = vectorizer
        self.alpha = alpha
        self.model = None
    
    def train(self, train_df, y_col, combined_text_col):
        """Train the Naive Bayes classifier."""
        X = train_df[combined_text_col]
        y = train_df[y_col]

        # Train Naive Bayes model
        self.model = make_pipeline(
            self.vectorizer(stop_words='english'), 
            MultinomialNB(alpha=self.alpha)
        )
        self.model.fit(X, y)
        return self.model
    
    def save_model(self, filepath='naive_bayes_model.joblib'):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath='naive_bayes_model.joblib'):
        """Load a pre-trained model."""
        self.model = joblib.load(filepath)
        return self.model
    
    def predict(self, test_df, combined_text_col, pred_col='Predicted_category'):
        """Make predictions on test data."""
        if self.model is None:
            self.load_model()
        
        # Get predictions and probabilities
        predicted_categories = self.model.predict(test_df[combined_text_col])
        predicted_probabilities = self.model.predict_proba(test_df[combined_text_col])

        # Add results to DataFrame
        test_df = test_df.copy()
        test_df[pred_col] = predicted_categories
        test_df['Prediction_probability'] = np.max(predicted_probabilities, axis=1)
        
        return test_df


def cross_validate_model(train_df, y_col, combined_text_col, model, cv=5):
    """Perform cross-validation on the model."""
    X = train_df[combined_text_col]
    y = train_df[y_col]
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores


def grid_search_hyperparameters(train_df, y_col, combined_text_col, vectorizer=TfidfVectorizer):
    """Perform grid search for hyperparameter tuning."""
    vectorizer_name = vectorizer.__name__.replace("Vectorizer", "").lower()

    # Pipeline with text preprocessing and model
    pipeline = Pipeline([
        (vectorizer_name, vectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])

    # Define hyperparameter grid
    param_grid = {
        'clf__alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5, 10, 100],
        f'{vectorizer_name}__max_features': [1000, 5000, 10000]
    }

    # Initialize grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Execute search
    grid_search.fit(train_df[combined_text_col], train_df[y_col])

    # Results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_params_


def print_classification_report(results_df, y_col, pred_col):
    """Print detailed classification report."""
    y_true = results_df[y_col]
    y_pred = results_df[pred_col]
    print(classification_report(y_true, y_pred))
