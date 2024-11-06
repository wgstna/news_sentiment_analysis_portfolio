import os
import numpy as np
from flask import Flask, render_template, request
import requests
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Use environment variable for API key
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')


def get_news_articles(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}&language=en&pageSize=100"
    response = requests.get(url)
    return response.json()['articles']


def analyze_sentiment(text):
    if not isinstance(text, str):
        logging.warning(f"Non-string input to analyze_sentiment: {type(text)}")
        return 0  # or some neutral sentiment value
    return TextBlob(text).sentiment.polarity

def categorize_sentiment(score):
    if not isinstance(score, (int, float)):
        logging.warning(f"Non-numeric input to categorize_sentiment: {type(score)}")
        return 'Neutral'
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def create_sentiment_plot(df):
    plt.figure(figsize=(8, 8))
    sentiment_counts = df['sentiment_category'].value_counts()
    colors = ['#a8e6cf', '#ffd3b6', '#ffaaa5']  # Pastel Green, Orange, Pink
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=colors, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64


import logging

logging.basicConfig(level=logging.DEBUG)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            topic = request.form['topic'].upper()
            articles = get_news_articles(topic)

            df = pd.DataFrame(articles)
            df['sentiment_score'] = df['description'].apply(analyze_sentiment)
            df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

            average_sentiment = df['sentiment_score'].mean()

            sentiment_counts = df['sentiment_category'].value_counts().to_dict()
            total_count = sum(sentiment_counts.values())
            sentiment_percentages = {k: f"{(v / total_count * 100):.1f}%" for k, v in sentiment_counts.items()}

            plot_base64 = create_sentiment_plot(df)

            return render_template('results.html',
                                   topic=topic,
                                   average_sentiment=average_sentiment,
                                   sentiment_counts=sentiment_counts,
                                   sentiment_percentages=sentiment_percentages,
                                   plot_base64=plot_base64)
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


