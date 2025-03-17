from utils import *
import emoji
import re
import spacy
from bs4 import BeautifulSoup

#Todo: test if emojis are kept after removing some non-alphanumeric characters

nlp = spacy.load("en_core_web_sm")

def preprocess_general(file_path):
    file_content = read_file(file_path)

    # Strip HTML/XML tags with BeautifulSoup
    file_content = BeautifulSoup(file_content, "html.parser").get_text()
    # Lower case
    file_content = file_content.lower()
    # Whitespace
    file_content = re.sub(r'\s+', ' ', file_content).strip()
    # Remove some non-alphanumeric characters
    file_content = re.sub(r'[^\w\s!.,;?\'":-]', '', file_content) 
    # Segment with spacy
    doc = nlp(file_content)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return sentences

def preprocess_sentiment_analysis(file_path):
    normalized_sentences = preprocess_general(file_path)




