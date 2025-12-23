from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load Model and Tokenizer globally for efficiency
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    # Convert logits to probabilities
    sentiment_scores = torch.softmax(logits, dim=1).tolist()[0]
    
    # Custom threshold logic as per your original code
    # Index 0: Negative, Index 1: Positive
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    sentiment, score = analyze_sentiment(text)
    return jsonify({'sentiment': sentiment, 'score': score})

if __name__ == '__main__':
    app.run(debug=True)