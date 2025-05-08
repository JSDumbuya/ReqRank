from transformers import pipeline
from preprocess import preprocess_sentiment_analysis

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def perform_sentiment_analysis(file_path, pipeline):
    preprocessed_content = preprocess_sentiment_analysis(file_path)

    results = [(data, pipeline(data)[0]) for data in preprocessed_content]
    return results

#For testing
file_path = "backend/testing/Test_sentiment_analysis.csv"
results = perform_sentiment_analysis(file_path, sentiment_pipeline)
print(results)

