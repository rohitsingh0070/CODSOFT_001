import pickle

# Load the trained model and vectorizer
def load_model():
    with open('../models/genre_classifier.pkl', 'rb') as model_file:
        model, vectorizer = pickle.load(model_file)
    return model, vectorizer

# Predict genre for a new plot summary
def predict_genre(plot_summary):
    model, vectorizer = load_model()

    # Transform the plot summary into TF-IDF features
    plot_summary_tfidf = vectorizer.transform([plot_summary])

    # Predict genre
    genre_pred = model.predict(plot_summary_tfidf)
    return genre_pred[0]

if __name__ == "__main__":
    # Example plot summary for prediction
    plot_summary = "A young wizard embarks on an epic journey to defeat a dark lord."
    predicted_genre = predict_genre(plot_summary)
    print(f"Predicted Genre: {predicted_genre}")
