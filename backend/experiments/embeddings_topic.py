from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from utils import *


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(sentences, model):
    embeddings = model.encode(sentences)
    return sentences, embeddings

def extract_topics(sentences, embeddings):
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(documents=sentences, embeddings=embeddings)
    return topic_model, topics, probabilities

#Testing
'''sentences, embeddings = generate_embeddings("backend/datasets/topic-Eval-Dataset.csv", embedding_model)
model, topics, probs = extract_topics(sentences, embeddings)
create_csv(model.get_topic_info(), "topics_text.csv")'''


