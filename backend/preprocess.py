from utils import read_file
import pandas as pd
import emoji
import re
import spacy
from bs4 import BeautifulSoup
from wtpsplit import SaT
import copy

sat_sm = SaT("sat-3l-sm")
nlp_base = spacy.load("en_core_web_md")
nlp_default = copy.deepcopy(nlp_base)
nlp_sentiment = copy.deepcopy(nlp_base)

def normalize(file_path):
    if isinstance(file_path, pd.DataFrame):
        file_content = file_path.to_string(index=False, header=False)
    else:
        file_content = read_file(file_path)
        if isinstance(file_content, pd.DataFrame):
            file_content = file_content.dropna(how='all')
            file_content = ' '.join(file_content.astype(str).values.flatten())
        else:
            file_content = str(file_content)


    # Strip HTML/XML tags: BeautifulSoup
    file_content = BeautifulSoup(file_content, "html.parser").get_text()
    # Lower case
    file_content = file_content.lower()
    # Remove trailing whitespace
    file_content = re.sub(r'\s+', ' ', file_content).strip()
    
    return file_content

def segment_sat(file_content):
    # Segment SaT
    sentences = sat_sm.split(file_content)
    return sentences

def preprocess_embeddings(file_path):
    if isinstance(file_path, pd.DataFrame):
        file_content = file_path
    else:
        file_content = read_file(file_path)
    
    if not isinstance(file_content, pd.DataFrame):
        file_content = pd.DataFrame([file_content], columns=["text"])
    
    normalized_content = normalize(file_content)
    segmented_sentences = segment_sat(normalized_content)
    return segmented_sentences

stopwords_to_keep = [
    "not", "no", "never", "n't", "nor", "neither", 
    "very", "much", "more", "most", "so", "too", "quite", "just", "really",  
    "would", "could", "should", "might", "must", "may", "can", "will", "do", "does", "did", 
    "but", "although", "however", "though", "yet", "still" 
]

for word in stopwords_to_keep:
    nlp_sentiment.vocab[word].is_stop = False

def preprocess_sentiment_analysis(file_path):
    normalized_content = normalize(file_path)
    segmented_sentences = segment_sat(normalized_content)

    # Emoji conversion: emoji
    segmented_sentences = [emoji.demojize(sentence) for sentence in segmented_sentences]
    # Lemmatization, stop word removal: spaCy
    processed_sentences = []
    for sentence in segmented_sentences:
        doc = nlp_sentiment(sentence)
        processed_sentence = " ".join([token.lemma_ for token in doc if not token.is_stop]).strip()
        processed_sentences.append(processed_sentence)

    return processed_sentences 

def preprocess_feedback(file_path):
    normalized_content = normalize(file_path)
    segmented_sentences = segment_sat(normalized_content)

    # Lemmatization, stop word removal: spaCy
    processed_sentences = []
    for sentence in segmented_sentences:
        doc = nlp_base(sentence)
        processed_sentence = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]).strip()
        processed_sentences.append(processed_sentence)

    return processed_sentences 

def normalize_lists(text):
    # Strip HTML/XML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lower case
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_lists(texts):
    normalized_content = [normalize_lists(text) for text in texts]

    normalized_content = [emoji.demojize(sentence) for sentence in normalized_content]

    processed_sentences = []
    for sentence in normalized_content:
        doc = nlp_base(sentence)
        processed_sentence = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]).strip()
        processed_sentences.append(processed_sentence)

    return processed_sentences

def normalize_testing(file_path):
    if isinstance(file_path, pd.DataFrame):
        file_content = file_path.to_string(index=False, header=False)
    else:
        file_content = read_file(file_path)

    if isinstance(file_content, pd.DataFrame):
        file_content = file_content.to_string(index=False, header=False)
    
    # Lower case
    file_content = file_content.lower()
    # Remove trailing whitespace
    file_content = re.sub(r'\s+', ' ', file_content).strip()

    return file_content

def preprocess_classification_experiments(file_path):
    if isinstance(file_path, pd.DataFrame):
        file_content = file_path.to_string(index=False, header=False)
    else:
        file_content = read_file(file_path)
    
    if isinstance(file_content, pd.DataFrame):
        file_content = file_content.to_string(index=False, header=False)

    #split into individual reqs
    lines = file_content.split("\n")
    
    cleaned_reqs = []

    for line in lines:
        # lower case, remove potential whitespace
        line = line.lower().strip()
        # Remove trailing whitespace
        line = re.sub(r'\s+', ' ', line).strip()
        # stop word, punctuation removal: spaCy
        doc = nlp_base(line)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        cleaned_req = " ".join(tokens).strip()
        if cleaned_req:
            cleaned_reqs.append(cleaned_req)

    return cleaned_reqs

def preprocess_reqs_clustering(file_path):
    if isinstance(file_path, pd.DataFrame):
        file_content = file_path.to_string(index=False, header=False)
    else:
        file_content = read_file(file_path)

    lines = file_content.split("\n")
    
    cleaned_reqs = []
    
    for line in lines:
        # Lowercase and strip whitespace
        line = line.lower().strip()
        line = re.sub(r'\s+', ' ', line).strip()

        # Lemmatize using spaCy
        doc = nlp_base(line)
        lemmatized_line = ' '.join(token.lemma_ for token in doc if not token.is_space)

        cleaned_reqs.append(lemmatized_line)

    return cleaned_reqs


def preprocess_classification_production(prepared_reqs):
    cleaned_reqs = []
    for req in prepared_reqs.values():
        line = req["text"].lower().strip()
        line = re.sub(r'\s+', ' ', line).strip()

        doc = nlp_base(line)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        cleaned_line = " ".join(tokens).strip()

        if cleaned_line:
            cleaned_reqs.append(cleaned_line)

    return cleaned_reqs   


def preprocess_reqs(file_path):
    if isinstance(file_path, pd.DataFrame):
        file_content = file_path.to_string(index=False, header=False)
    else:
        file_content = read_file(file_path)

    lines = file_content.split("\n")
    
    cleaned_reqs = []
    
    for line in lines:
        # lower case, remove potential whitespace
        line = line.lower().strip()
        # Remove trailing whitespace
        line = re.sub(r'\s+', ' ', line).strip()
        cleaned_reqs.append(line)

    return cleaned_reqs

#For inspection of stopword list:
#print(sorted(list(nlp_base.Defaults.stop_words)))