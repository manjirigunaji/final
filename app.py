# app.py

import joblib
from flask import Flask, request, jsonify
import streamlit as st

# Load the trained model
model = joblib.load('sentiment_analysis_model.pkl')

# Flask App
app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    reviews = data['reviews']
    predictions = model.predict(reviews)
    return jsonify(predictions.tolist())

# Streamlit Frontend
def main():
    st.title("Sentiment Analysis Demo")
    st.write("Enter movie reviews to get sentiment predictions!")

    reviews = st.text_area("Enter reviews (one per line):", "", height=200)

    if st.button("Predict Sentiments"):
        reviews_list = reviews.split('\n')
        predictions = model.predict(reviews_list)

        st.write("\nPredictions:")
        for review, prediction in zip(reviews_list, predictions):
            st.write(f"Review: '{review}'\nSentiment: {prediction}\n")

if __name__ == '__main__':
    main()
