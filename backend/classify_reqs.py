import pandas as pd
import pickle
from preprocess import preprocess_classification
from sentence_transformers import SentenceTransformer

with open("svm_model_binary.pkl", "rb") as model_file:
    svm_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("rbf_model_multi.pkl", "rb") as rbf_model_file:
    rbf_svm_model = pickle.load(rbf_model_file)

def binary_classification(file_path):
    cleaned_reqs = preprocess_classification(file_path)
    cleaned_reqs_df = pd.DataFrame({'cleaned_reqs': cleaned_reqs})

    X_new = vectorizer.transform(cleaned_reqs_df['cleaned_reqs'])
    predictions = svm_model.predict(X_new)
    
    fr_list = [req for req, label in zip(cleaned_reqs, predictions) if label == "FR"]
    nfr_list = [req for req, label in zip(cleaned_reqs, predictions) if label == "NFR"]

    return fr_list, nfr_list
    
def multi_class_classification(file_path):
    cleaned_reqs = preprocess_classification(file_path)
    cleaned_reqs_df = pd.DataFrame({'cleaned_reqs': cleaned_reqs})

    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_new_embeddings = model.encode(cleaned_reqs_df['cleaned_reqs'].tolist(), show_progress_bar=True)
    predictions = rbf_svm_model.predict(X_new_embeddings)

    result_dict = {}
    for label in set(predictions): 
        result_dict[label] = [req for req, pred in zip(cleaned_reqs, predictions) if pred == label]

    return result_dict