# app.py (Flask back-end)

from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the sentiment analysis models and tokenizers
models = {
    "distilbert": {
        "tokenizer": DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        "model": DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    },
    "tfidf_lr": {
        "vectorizer": joblib.load('tfidf_vectorizer.sav'),
        "model": joblib.load('trained_model.sav')  # Updated model file name
    }
    # Add more models here if needed
}

# Define a function to perform sentiment analysis using the specified model
def analyze_sentiment(text, selected_model):
    model_info = models[selected_model]
    if selected_model == "distilbert":
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Perform sentiment analysis
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        labels = ['negative', 'positive']
        
        sentiment_label = labels[predicted_class]
        sentiment_scores = torch.softmax(logits, dim=1).tolist()[0]
        
        if sentiment_scores[0] <= 0.10 and sentiment_scores[1] >= 0.65:
            sentiment_label = 'Positive'
            score = sentiment_scores[1]
        elif sentiment_scores[0] >= 0.65 and sentiment_scores[1] <= 0.10:
            sentiment_label = 'Negative'
            score = sentiment_scores[0]
        else:
            sentiment_label = 'Neutral'
            score = max(sentiment_scores)
        
        return sentiment_label, score
    elif selected_model == "tfidf_lr":
        vectorizer = model_info["vectorizer"]
        model = model_info["model"]
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)
        if prediction == 1:
            return 'Positive', 1
        elif prediction == -1:
            return 'Negative', -1
        else:
            return 'Neutral', 0
    else:
        return 'Model not implemented', 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    selected_model = request.form['model']
    # Perform sentiment analysis using the selected model
    sentiment, score = analyze_sentiment(text, selected_model)
    return jsonify({'sentiment': sentiment, 'score': score})

if __name__ == '__main__':
    app.run(debug=True)
