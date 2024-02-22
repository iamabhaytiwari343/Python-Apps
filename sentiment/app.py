import streamlit as st
from textblob import TextBlob

# Streamlit app title
st.title("Sentiment Analyzer App")

# User input for text
user_input = st.text_area("Enter the text you want to analyze:")

# Sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        # Perform sentiment analysis using TextBlob
        analysis = TextBlob(user_input)
        
        # Display the sentiment polarity and subjectivity
        st.write("Sentiment Analysis Results:")
        st.write(f"Text: {user_input}")
        st.write(f"Sentiment Polarity: {analysis.sentiment.polarity}")
        st.write(f"Sentiment Subjectivity: {analysis.sentiment.subjectivity}")

        # Display sentiment labels
        sentiment_label = "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"
        st.write(f"Sentiment Label: {sentiment_label}")

    else:
        st.warning("Please enter some text for analysis.")
