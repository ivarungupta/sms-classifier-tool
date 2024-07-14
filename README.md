# SMS Classifier Tool

A Flask web application that classifies SMS messages as spam or ham using a Naive Bayes model. This project utilizes machine learning techniques to preprocess text data, train a classification model, and provide a user-friendly interface for message classification.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ivarungupta/sms-classifier-tool.git
    cd sms-classifier-tool
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask web application:
    ```bash
    python app.py
    ```

2. Open your browser and go to `http://127.0.0.1:5001` to access the application.

3. Enter a message in the text area and click "Classify" to determine whether the message is spam or ham.

## Dataset

The dataset used in this project is the SMS Spam Collection dataset, which contains a collection of SMS messages labeled as spam or ham.

## Model Training

The model is trained using a Naive Bayes classifier with text data preprocessed and vectorized using the `CountVectorizer` from scikit-learn. The following steps are performed:

1. Data preprocessing: Convert text to lowercase, remove special characters, extra spaces, and digits.
2. Data splitting: Split the dataset into training and testing sets.
3. Vectorization: Convert text data into numerical features using `CountVectorizer`.
4. Model training: Train a Naive Bayes classifier on the training data.
5. Model evaluation: Evaluate the model's performance on the test data.

## Web Application

The web application is built using Flask and provides a simple interface to classify SMS messages. It consists of two main routes:

- `/`: Displays the home page with a form to enter and classify a message.
- `/classify`: Processes the form input, classifies the message, and displays the result.

## Results

The model achieves an accuracy of approximately 98% on the test dataset. Detailed classification report and accuracy score are provided in the console output.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to improve the project.

## Thanks!!!

