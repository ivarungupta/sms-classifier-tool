import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib

def preprocess_text(text):
    """Preprocess the text data."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

def main():
    csv_path = "/Users/srijitaseth/sms_classifier /sms_spam_collection (1).csv"

    # Load dataset
    df = pd.read_csv(csv_path, encoding='utf-8')  # Updated to utf-8

    print("First few rows of the dataset:")
    print(df.head())

    # Ensure correct label encoding
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    # Preprocess text
    df['text'] = df['text'].apply(preprocess_text)

    print("Label distribution:")
    print(df['label'].value_counts())

    if len(df['label'].unique()) < 2:
        print("Dataset contains only one class. Ensure that the dataset contains both 'ham' and 'spam' messages.")
        return

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    print("Training set label distribution:")
    print(y_train.value_counts())
    print("Test set label distribution:")
    print(y_test.value_counts())

    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(solver='liblinear')),
    ])

    # Define the parameter grid
    param_grid = [
        {
            'tfidf__max_df': [0.8, 0.9, 1.0],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf': [LogisticRegression(solver='liblinear')],
            'clf__C': [0.1, 1, 10],
        },
        {
            'tfidf__max_df': [0.8, 0.9, 1.0],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf': [MultinomialNB()],
            'clf__alpha': [0.1, 1, 10],
        },
        {
            'tfidf__max_df': [0.8, 0.9, 1.0],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf': [RandomForestClassifier()],
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [10, 30, 50, None],
            'clf__class_weight': [None, 'balanced'],
        },
    ]

    # Perform GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found by grid search:")
    print(grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))

    # Save the model
    joblib.dump(best_model, 'sms_spam_classifier.pkl')
    print("Model saved as 'sms_spam_classifier.pkl'")

if __name__ == "__main__":
    main()
