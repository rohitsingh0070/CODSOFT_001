
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(file_path):

    data = pd.read_csv(file_path, encoding='latin-1')  
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    return data

def preprocess_data(data):
    
    data['label'] = data['label'].map({'spam': 1, 'ham': 0})
    return data

def train_model(X_train, y_train):
  
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
   
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    return model, tfidf

def evaluate_model(model, tfidf, X_test, y_test):
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():

    data = load_data('c:\\Users\\royal\\Downloads\\archive (4)\\spam.csv')

    data = preprocess_data(data)
    

    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=42)

    model, tfidf = train_model(X_train, y_train)
    
    evaluate_model(model, tfidf, X_test, y_test)

if __name__ == "__main__":
    main()
