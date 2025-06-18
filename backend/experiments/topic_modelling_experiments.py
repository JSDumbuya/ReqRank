from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import gensim
from transformers import DistilBertTokenizer, DistilBertModel
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import torch
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import hdbscan

#Trying HBDSCAN and LDA + distilbert embeddings for topic modelling setup on 'small' datasets

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_topic_keywords(lda_model, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-top_n - 1:-1]]
        topic_keywords.append(", ".join(top_features))
    return topic_keywords

def get_distilbert_embeddings(sentences):
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy() 
        embeddings.append(sentence_embedding)
    return np.array(embeddings)

def compare_clustering_methods(all_sentences, all_embeddings):
    #all_distil_embeddings = get_distilbert_embeddings(all_sentences)

    # HDBSCAN (silhouette score)
    cosine_distance_matrix = cosine_distances(all_embeddings)
    cosine_distance_matrix = cosine_distance_matrix.astype(np.float64)

    hdbscan_clusterer = hdbscan.HDBSCAN(metric='precomputed')
    hdbscan_labels = hdbscan_clusterer.fit_predict(cosine_distance_matrix)
    hdbscan_data = [{'Sentence': all_sentences[i], 'Cluster': label} for i, label in enumerate(hdbscan_labels)]
    hdbscan_df = pd.DataFrame(hdbscan_data)
    hdbscan_df.to_csv('hdbscan_results.csv', index=False)

    # LDA (coherence score)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
    lda_topic_matrix = lda_model.fit_transform(tfidf_matrix)
    lda_labels = lda_topic_matrix.argmax(axis=1)
    topic_keywords = get_topic_keywords(lda_model, vectorizer)

    '''lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_topic_matrix = lda_model.fit_transform(all_distil_embeddings)  # LDA on DistilBERT embeddings
    lda_labels = lda_topic_matrix.argmax(axis=1)'''

    lda_data = []
    for i, label in enumerate(lda_labels):
        lda_data.append({
            'Sentence': all_sentences[i],
            'Topic': label,
            'Topic Keywords': topic_keywords[label]
        })
    lda_df = pd.DataFrame(lda_data)
    
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in all_sentences]
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]

    gensim_lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,
        passes=10,
        random_state=42
    )

    coherence_model_lda = CoherenceModel(
        model=gensim_lda,
        texts=tokenized_sentences,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model_lda.get_coherence()

    print(f"LDA Topic Coherence (c_v): {coherence_score:.3f}")

    lda_df['Topic Coherence'] = coherence_score
    lda_df.to_csv('lda_results.csv', index=False)

    if len(set(hdbscan_labels)) > 1 and -1 not in set(hdbscan_labels):
        sil_score = silhouette_score(all_embeddings, hdbscan_labels)
        print(f"HDBSCAN Silhouette Score: {sil_score:.3f}")
    else:
        print("HDBSCAN Silhouette Score: Not available (only one cluster or too much noise)")

    return hdbscan_df, lda_df