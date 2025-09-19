import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

import streamlit as st

## Loading the words in imdb data set
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

##Load the Trained model
model = load_model("simple_rnn_imdb.h5")

## Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen = 500)
    return padded_review


## Prediction of the review.
def predict_sentiment(review):
    padded_review = preprocess_text(review)
    prediction = model.predict(padded_review)
    sentiment = "Positive" if prediction[0][0] else "Negative"
    return sentiment,prediction[0][0]

## Streamlit app
st.title("Welcome to IMDB Movie Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative,")

## User Input
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    sentiment,prediction = predict_sentiment(user_input)
    ## Display
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Prediction:{prediction}")
else:
    st.write("Please enter a movie review.")