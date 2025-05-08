import pandas as pd
import pickle
from preprocess import preprocess_classification_production

with open("models/svm_model_binary.pkl", "rb") as model_file:
    svm_model = pickle.load(model_file)

with open("models/tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("models/rbf_model_multi.pkl", "rb") as rbf_model_file:
    rbf_svm_model = pickle.load(rbf_model_file)

def binary_classification(prepared_reqs):
    cleaned_reqs = preprocess_classification_production(prepared_reqs)
    cleaned_reqs_df = pd.DataFrame({'cleaned_reqs': cleaned_reqs})
    req_ids = list(prepared_reqs.keys())

    X_new = vectorizer.transform(cleaned_reqs_df['cleaned_reqs'])
    predictions = svm_model.predict(X_new)

    fr_ids = [req_id for req_id, label in zip(req_ids, predictions) if label == "FR"]
    nfr_ids = [req_id for req_id, label in zip(req_ids, predictions) if label == "NFR"]

    return fr_ids, nfr_ids

def multi_class_classification(prepared_reqs):
    nfr_ids = [req_id for req_id, req_info in prepared_reqs.items() if req_info.get("type") == "NFR"]

    embeddings = [prepared_reqs[req_id]["embedding"] for req_id in nfr_ids]

    if not embeddings:
        return {}

    category_preds = rbf_svm_model.predict(embeddings)

    category_ids = {category: [] for category in set(category_preds)}

    for req_id, category in zip(nfr_ids, category_preds):
        category_ids[category].append(req_id)

    return category_ids
