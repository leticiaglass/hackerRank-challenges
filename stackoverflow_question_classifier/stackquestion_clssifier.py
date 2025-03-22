import json
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Load and preprocess the training data
def load_training_data(file_path):
    with open(file_path, 'r') as f:
        num_lines = int(f.readline().strip())
        data = [json.loads(f.readline().strip()) for _ in range(num_lines)]
    df = pd.DataFrame(data)
    return df

# Preprocess text
def preprocess_text(df):
    df['text'] = df['question'] + " " + df['excerpt']
    return df

# Encode labels
def encode_labels(df):
    le = LabelEncoder()
    df['topic_encoded'] = le.fit_transform(df['topic'])
    return df, le

# Vectorize text
def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    return X, vectorizer

# Step 2: Train the model
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Step 3: Predict on input data
def predict_input_data(model, vectorizer, le):
    # Read input from stdin
    input_data = sys.stdin.read().splitlines()
    N = int(input_data[0])  # First line is the number of JSON objects
    test_data = [json.loads(line) for line in input_data[1:N+1]]  # Read N JSON objects
    test_df = pd.DataFrame(test_data)
    test_df['text'] = test_df['question'] + " " + test_df['excerpt']
    X_test = vectorizer.transform(test_df['text'])
    predictions = model.predict(X_test)
    predicted_topics = le.inverse_transform(predictions)
    return predicted_topics

# Main pipeline
def main():
    # Load and preprocess training data
    train_file_path = 'training.json'
    df = load_training_data(train_file_path)
    df = preprocess_text(df)
    df, le = encode_labels(df)
    X, vectorizer = vectorize_text(df)

    # Train the model
    model = train_model(X, df['topic_encoded'])

    # Predict on input data and print results
    predicted_topics = predict_input_data(model, vectorizer, le)
    for topic in predicted_topics:
        print(topic)

if __name__ == "__main__":
    main()