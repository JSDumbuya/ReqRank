from utils import read_file
import pandas as pd
import emoji
import re
import spacy
from bs4 import BeautifulSoup
from wtpsplit import SaT

sat_sm = SaT("sat-3l-sm")

def normalize(file_path):
    file_content = read_file(file_path)

    if isinstance(file_content, pd.DataFrame):
        file_content = file_content.to_string(index=False, header=False)

    # Strip HTML/XML tags: BeautifulSoup
    file_content = BeautifulSoup(file_content, "html.parser").get_text()
    # Lower case
    file_content = file_content.lower()
    # Whitespace
    file_content = re.sub(r'\s+', ' ', file_content).strip()

    return file_content

def segment_sat(file_content):
    #Segment SaT
    sentences = sat_sm.split(file_content)
    return sentences

def preprocess_embeddings(file_path):
    normalized_content = normalize(file_path)
    segmented_sentences = segment_sat(normalized_content)
    return segmented_sentences
    

nlp = spacy.load("en_core_web_sm")

def preprocess_sentiment_analysis(file_path):
    normalized_content = normalize(file_path)
    segmented_sentences = segment_sat(normalized_content)

    # Emoji conversion: emoji
    segmented_sentences = [emoji.demojize(sentence) for sentence in segmented_sentences]
    # Lemmatization, stop word removal: spaCy
    processed_sentences = []
    for sentence in segmented_sentences:
        doc = nlp(sentence)
        processed_sentence = " ".join([token.lemma_ for token in doc if not token.is_stop]).strip()
        processed_sentences.append(processed_sentence)

    return processed_sentences 

def preprocess_reqs(file_path):
    file_content = read_file(file_path)

    if isinstance(file_content, pd.DataFrame):
        file_content = file_content.to_string(index=False, header=False)

    #split into individual reqs
    lines = file_content.split("\n")
    
    cleaned_reqs = []

    for line in lines:
        # lower case, remove potential whitespace
        line = line.lower().strip()
        # stop word, punctuation removal: spaCy
        doc = nlp(line)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        cleaned_req = " ".join(tokens).strip()
        if cleaned_req:
            cleaned_reqs.append(cleaned_req)

    return cleaned_reqs


'''#Testing pipeline
file_path = "backend/testing/test_preprocess.txt"
test_general = preprocess_general(file_path)
test_sentiment = preprocess_sentiment_analysis(file_path)
create_csv(test_sentiment, "general_sent")'''

'''
Spacy stopword list
for stopword in nlp.Defaults.stop_words:
    print(stopword)
'''