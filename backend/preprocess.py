from utils import *
import emoji
import re
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")

def preprocess_general(file_path):
    file_content = read_file(file_path)

    # Strip HTML/XML tags: BeautifulSoup
    file_content = BeautifulSoup(file_content, "html.parser").get_text()
    # Lower case
    file_content = file_content.lower()
    # Whitespace
    file_content = re.sub(r'\s+', ' ', file_content).strip()
    # Remove some non-alphanumeric characters
    #file_content = re.sub(r'[^\w\s!.,;?\'":-]', '', file_content) 
    # Segment: spaCy
    doc = nlp(file_content)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return sentences

def preprocess_sentiment_analysis(file_path):
    normalized_sentences = preprocess_general(file_path)

    #Emoji conversion: emoji
    normalized_sentences = [emoji.demojize(sentence) for sentence in normalized_sentences]
    #Lemmatization, stop word removal: spaCy
    processed_sentences = []
    for sentence in normalized_sentences:
        doc = nlp(sentence)
        processed_sentence = " ".join([token.lemma_ for token in doc if not token.is_stop]).strip()
        processed_sentences.append(processed_sentence)

    return processed_sentences 



'''#Testing pipeline
file_path = "backend/uploads/test_preprocess.txt"
test_general = preprocess_general(file_path)
test_sentiment = preprocess_sentiment_analysis(file_path)
create_csv(test_sentiment, "general_sent")'''

'''
Spacy stopword list
for stopword in nlp.Defaults.stop_words:
    print(stopword)
'''