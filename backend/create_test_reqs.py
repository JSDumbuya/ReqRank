from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from preprocess import * 
#import pandas as pd

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

'''df = pd.read_csv('backend/datasets/capterra_outlook_reviews.csv')
review_texts = df[['review_text']]
review_texts.to_csv('backend/datasets/extracted_review_texts.csv', index=False)'''

review_texts_file_path = "backend/datasets/capterra_review_texts.csv"

preprocessed_review_data = preprocess_sentiment_analysis(review_texts_file_path)
review_embeddings = embedding_model.encode(preprocessed_review_data)
topic_model = BERTopic(min_topic_size=4)
topics, _ = topic_model.fit_transform(preprocessed_review_data, review_embeddings)

topic_info_df = topic_model.get_topic_info()
topic_info_df.to_csv("backend/datasets/outlook_reviews_topics.csv", index=False)

