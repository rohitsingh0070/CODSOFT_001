import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle

# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Assuming the dataset has columns 'plot' for movie plot summaries and 'genre' for genres
    X = data['plot']  # Movie plot summaries
    y = data['genre']  # Genres
    return X, y

# Train the model
def train_model():
    # Load dataset
    X, y = load_data(r'C:\Users\royal\Downloads\archive (5)\movies_metadata.csv')

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert plot summaries into TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Choose the classifier: Naive Bayes, Logistic Regression, or SVM
    model_choice = 'logistic_regression'  # Change to 'naive_bayes' or 'svm' if you prefer

    if model_choice == 'naive_bayes':
        model = MultinomialNB()
    elif model_choice == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    else:
        model = SVC(kernel='linear')

    # Train the model
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    # Save the trained model and vectorizer
    with open('../models/genre_classifier.pkl', 'wb') as model_file:
        pickle.dump((model, vectorizer), model_file)

if __name__ == "__main__":
    train_model()
