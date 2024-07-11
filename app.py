from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re

app = Flask(__name__)

CSV_FILE_PATH = '/Users/srijitaseth/sms_classifier /sms_spam_collection (1).csv'

df = pd.read_csv(CSV_FILE_PATH)

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

df['text'] = df['text'].apply(preprocess_text)

# Check the data
print(df.head())
print("Unique labels in the dataset:", df['label'].unique())

df['label'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Check vectorizer output
print(f"Number of train samples: {X_train_vectorized.shape[0]}")
print(f"Number of test samples: {X_test_vectorized.shape[0]}")

# Check class distribution
print(f"Class distribution in training set: {y_train.value_counts()}")

# Model Training
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Check training accuracy
train_pred = model.predict(X_train_vectorized)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Model Evaluation
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Function to classify messages
def classify_message(message):
    message = preprocess_text(message)  # Preprocess the message
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return 'Spam' if prediction[0] == 1 else 'Ham'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    message = request.form.get('message')
    if message:
        result = classify_message(message)
        print(f"Classified Message: {message} -> {result}")
        return render_template('result.html', message=message, result=result)
    return redirect(url_for('index'))

if __name__ == '__main__':
    test_message = "Congratulations! You've won a $1000 gift card. Click here to claim your prize!"
    result = classify_message(test_message)
    print(f"Test Message: {test_message} -> {result}")

    app.run(port=5001, debug=True)
