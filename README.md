# Ticket Classifier

A machine learning-based ticket classification system that automatically categorizes support tickets using Natural Language Processing (NLP) techniques. The model achieves an **85% accuracy rate** in predicting ticket categories on the Kaggle IT Tickets Data set.

## 🎯 Project Overview

This project implements a text classification pipeline to automatically categorize IT support tickets based on their descriptions. The system uses supervised machine learning to learn patterns from historical ticket data and predict categories for new tickets.

## 📊 Performance Metrics

- **Accuracy**: 85%
- **Model Type**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Cross-validation**: 5-fold stratified validation

## 🔬 Scientific Methods & Approach

### 1. Data Preprocessing
- **Text Combination**: Merges `short_description` and `description` fields for richer feature representation
- **Stratified Splitting**: 80/20 train-test split maintaining class distribution

### 2. Feature Engineering
- **Text Vectorization**: TF-IDF vectorization with English stop-word removal
- **Feature Selection**: Configurable max_features parameter (tuned between 1,000-10,000)
- **N-gram Analysis**: Unigram features for computational efficiency

### 3. Model Selection & Training
- **Algorithm**: Multinomial Naive Bayes
  - Chosen for its effectiveness with text classification
  - Handles sparse feature matrices efficiently
  - Provides probabilistic outputs for confidence scoring
- **Hyperparameter Tuning**: Grid search optimization for:
  - Alpha (smoothing parameter): 0.001 to 100
  - Max features: 1,000 to 10,000 features

### 4. Model Validation
- **Cross-Validation**: 5-fold stratified cross-validation
- **Performance Metrics**: Classification report including precision, recall, and F1-scores
- **Confidence Scoring**: Prediction probabilities for uncertainty quantification

### 5. Evaluation Framework
- **Train-Test Split**: Stratified sampling to maintain class distributions
- **Confusion Matrix Analysis**: Detailed per-class performance evaluation
- **Statistical Significance**: Minimum 10 samples per class requirement

## 🛠️ Technical Stack

- **Python 3.x**
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Joblib**: Model serialization
- **Matplotlib**: Visualization (word clouds)
- **WordCloud**: Text visualization

## 📁 Project Structure

```
ticket-classifier/
├── classifier.py           # Main classification pipeline
├── naive_bayes_model.joblib # Trained model (generated)
├── output_with_correct.xlsx # Test results with accuracy flags
├── summary.xlsx            # Performance summary
├── data/
│   └── all_tickets.xlsx    # Input dataset
└── README.md              # This file
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install pandas scikit-learn numpy matplotlib wordcloud openpyxl joblib
   ```

2. **Prepare Data**
   - Place your ticket data in `data/all_tickets.xlsx`
   - Ensure columns: `sys_created_on`, `category`, `short_description`, `description`

3. **Run Classification**
   ```bash
   python classifier.py
   ```

4. **View Results**
   - `summary.xlsx`: Overall accuracy metrics
   - `output_with_correct.xlsx`: Detailed predictions with correctness flags

## 📈 Model Performance Analysis

### Cross-Validation Results
The model undergoes rigorous 5-fold cross-validation to ensure generalizability and avoid overfitting.

### Hyperparameter Optimization
Grid search optimization explores:
- **Alpha values**: Controls Laplace smoothing (prevents zero probabilities)
- **Feature counts**: Balances model complexity vs. performance
- **Vectorization parameters**: Optimizes text representation

### Statistical Rigor
- **Stratified sampling**: Maintains class distribution across train/test splits
- **Minimum sample requirements**: Ensures statistical significance
- **Probability scoring**: Provides confidence measures for predictions

## 🔍 Key Features

1. **Automated Hyperparameter Tuning**: Grid search finds optimal parameters
2. **Cross-Validation**: Robust performance estimation
3. **Confidence Scoring**: Probability estimates for each prediction
4. **Comprehensive Reporting**: Detailed classification reports and summaries
5. **Model Persistence**: Trained models saved for reuse
6. **Data Quality Filters**: Automatic filtering of low-frequency categories

## 📊 Output Files

- **`summary.xlsx`**: High-level accuracy statistics
- **`output_with_correct.xlsx`**: Test set with predictions and correctness flags
- **`naive_bayes_model.joblib`**: Serialized trained model

## 🧪 Experimental Design

The classification pipeline follows established machine learning best practices:

1. **Data Quality Assurance**: Temporal filtering and class balance validation
2. **Feature Engineering**: TF-IDF vectorization for optimal text representation
3. **Model Selection**: Naive Bayes chosen for interpretability and efficiency
4. **Hyperparameter Optimization**: Systematic grid search for optimal parameters
5. **Performance Validation**: Cross-validation and holdout testing
6. **Results Documentation**: Comprehensive reporting and model persistence

## 🔬 Scientific Validation

The 85% accuracy rate is validated through:
- **Stratified cross-validation**: Ensures results generalize across data splits
- **Holdout testing**: Independent test set for unbiased performance estimation
- **Statistical significance**: Minimum sample requirements prevent overfitting to small classes
- **Reproducible methodology**: Fixed random seeds and documented parameters

## 🚧 Future Enhancements

- **Deep Learning Models**: BERT/transformer-based approaches
- **Ensemble Methods**: Random Forest or Gradient Boosting combinations
- **Active Learning**: Iterative improvement with human feedback
- **Real-time Processing**: Streaming classification capabilities
- **Multi-label Classification**: Support for tickets with multiple categories

## 📝 License

This project is available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

---

*This classifier demonstrates the effective application of classical machine learning techniques to real-world text classification problems, achieving solid performance through rigorous scientific methodology.*
