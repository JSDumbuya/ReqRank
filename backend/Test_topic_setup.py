from bertopic import BERTopic
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from preprocess import *
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
import nltk
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import pandas as pd
#nltk.download('punkt')

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def tokenize_texts(texts):
    return [word_tokenize(text) for text in texts]

def create_features_embeddings(lines):
        preprocessed_feedback = preprocess_feedback(lines)
        preprocessed_feedback = [doc for doc in preprocessed_feedback if len(doc.split()) >= 4]
        print(len(preprocessed_feedback))
        
        vectorizer = None
  
        if len(preprocessed_feedback) >= 3000:
             embeddings = embedding_model.encode(preprocessed_feedback, show_progress_bar=False)
             representation_type = "bertopic"
        else:
            vectorizer = CountVectorizer(ngram_range=(1, 3))
            embeddings = vectorizer.fit_transform(preprocessed_feedback)
            representation_type = "lda"
            
        return representation_type, embeddings, preprocessed_feedback, vectorizer

def derive_topics(representation_type, all_embeddings, all_sentences, vectorizer):
    if representation_type == "bertopic":
        all_embeddings = np.array(all_embeddings)
        umap_model = UMAP(n_neighbors=15, n_components=15, metric='cosine', random_state=42)
        if len(all_sentences) < 3000:
            topic_size = 15
        elif len(all_sentences) < 10000:
            topic_size = 30
        else: 
            topic_size = 50
        topic_model = BERTopic(umap_model=umap_model, min_topic_size=topic_size)
        topics, _ = topic_model.fit_transform(all_sentences, all_embeddings)

    elif representation_type == "lda":
        doc_term_matrix = vectorizer.fit_transform(all_sentences)
        num_sentences = len(all_sentences)
        #30, 8
        #6, 50
        n_components = max(3, round(num_sentences // 2.5))
        # 8
        lda_model = LatentDirichletAllocation(n_components=n_components, random_state=42)
        doc_topics = lda_model.fit_transform(doc_term_matrix)
        topics = np.argmax(doc_topics, axis=1)
        topic_model = lda_model

    return topic_model, topics


def calculate_coherence(topic_model, representation_type, raw_sentences, vectorizer=None):
    tokenized_texts = tokenize_texts(raw_sentences)
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    '''if representation_type == "lda":
        topics = []
        n_top_words = 10
        for topic_idx, topic in enumerate(topic_model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [vectorizer.get_feature_names_out()[i] for i in top_features_ind]
            topics.append(top_features)'''
    if representation_type == "lda":
        topics = []
        n_top_words = 10
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(topic_model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = []
            for i in top_features_ind:
                if 0 <= i < len(feature_names):
                    tokens = feature_names[i].split()
                    top_features.extend(tokens)
            if top_features:
                topics.append(top_features)


    elif representation_type == "bertopic":
        topics_dict = topic_model.get_topics()
        topics = []
        n_top_words = 10
        for topic_num in topics_dict:
            if topic_num == -1:
                continue
            words_and_scores = topics_dict[topic_num]
            top_words = [word for word, _ in words_and_scores[:n_top_words]]
            topics.append(top_words)
    
    print("DEBUG: Sample topic structure:", topics[:1])
    print("DEBUG: Type of topics:", type(topics))
    print("DEBUG: Type of first topic:", type(topics[0]) if topics else "No topics")
    print("DEBUG: First topic content:", topics[0] if topics else "No topics")

    coherence_model = CoherenceModel(topics=topics, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    all_words = [word for topic in topics for word in topic]
    unique_words = set(all_words)
    topic_diversity = len(unique_words) / len(all_words)

    return coherence_score, topic_diversity

def load_dataset(file_path):
    return file_path


def test_topic_models(texts, dataset_name, output_csv="topics_output.csv"):
    rep_type, embeddings, preprocessed_sa, vectorizer = create_features_embeddings(texts)
    model, topics = derive_topics(rep_type, embeddings, preprocessed_sa, vectorizer)
    coherence, diversity = calculate_coherence(model, rep_type, preprocessed_sa, vectorizer)

    print(f"Dataset: {dataset_name}")
    print(f"Model used: {rep_type}")
    print(f"Topic coherence (c_v): {coherence:.4f}")
    print(f"Topic Diversity: {diversity:.4f}\n")

    rows = []
    if rep_type == "lda":
        for idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[:-11:-1]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
            rows.append({
                "dataset": dataset_name,
                "model": rep_type,
                "topic_num": idx,
                "top_words": ', '.join(top_words),
                "coherence": coherence,
                "diversity": diversity
            })
    elif rep_type == "bertopic":
        topics_dict = model.get_topics()
        for topic_num, words_scores in topics_dict.items():
            if topic_num == -1:
                continue
            top_words = [word for word, _ in words_scores[:10]]
            rows.append({
                "dataset": dataset_name,
                "model": rep_type,
                "topic_num": topic_num,
                "top_words": ', '.join(top_words),
                "coherence": coherence,
                "diversity": diversity
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, mode='a', index=False, header=not pd.io.common.file_exists(output_csv))

dataset_paths = {
    'capterra_reviews': 'backend/datasets/capterra_review_texts.txt',
    'lda_200': 'backend/datasets/lda_200_dataset.csv',
    'lda_450': 'backend/datasets/lda_450_dataset.csv',
    'bertopic_500': 'backend/datasets/bertopic_500_dataset.csv',
    'bertopic_700': 'backend/datasets/bertopic_700_dataset.csv',
    'bertopic_1000': 'backend/datasets/bertopic_1000_dataset.csv',
    'bertopic_2500': 'backend/datasets/bertopic_2500_dataset.csv',
    'bertopic_5000': 'backend/datasets/bertopic_5000_dataset.csv',
}

#bertopic_10000': 'backend/datasets/bertopic_10000_dataset.csv

if __name__ == "__main__":
    for name, path in dataset_paths.items():
        lines = load_dataset(path)
        test_topic_models(lines, name)


#Create files
''''
input_path = 'backend/datasets/sentiment-analysis-dataset-google-play-app-reviews.csv'
df = pd.read_csv(input_path)

contents = df['content'].dropna().reset_index(drop=True)
contents = contents.sample(frac=1, random_state=42).reset_index(drop=True)

dataset_sizes = {
    'lda_200': 200,
    'lda_450': 450,
    'bertopic_500': 500,
    'bertopic_700': 700,
    'bertopic_1000': 1000
}

for name, size in dataset_sizes.items():
    if size > len(contents):
        print(f"Warning: requested size {size} exceeds dataset length {len(contents)}")
        size = len(contents)
    subset = contents.iloc[:size]
    output_path = f'backend/datasets/{name}_dataset.csv'
    subset.to_csv(output_path, index=False)
    print(f"Saved {size} samples to {output_path}")
'''


