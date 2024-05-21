from flask import Flask, request, jsonify
import neattext.functions as nfx
import joblib
import numpy as np

app = Flask(__name__)

def load_files():
    global model, tfidf_vectorizer
    # Load the model
    model = joblib.load('naive_bayes_model.pkl')

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load files only once during app initialization
load_files()

def clean_text(text):
    cleaned_text = nfx.remove_special_characters(text)  # Remove special characters
    cleaned_text = nfx.remove_stopwords(text)  # Remove stopwords
    # Additional preprocessing steps can be added here
    return cleaned_text

@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        texts = data.get('texts', [])
        predictions = []
        predictions_score = []

        for text in texts:
            cleaned_text = clean_text(text)
            X_test = tfidf_vectorizer.transform([cleaned_text])
            y_probs = model.predict_proba(X_test)[:, 1]
            predictions.append(y_probs[0])
            predictions_score.append(y_probs[0])

        # Convert numpy arrays to lists and ensure they are serializable
        predictions = [float(pred) for pred in predictions]
        predictions_score = [float(score) for score in predictions_score]

        return jsonify({"predicted_labels": predictions, "predicted_scores": predictions_score})

if __name__ == '__main__':
    app.run(debug=False)
