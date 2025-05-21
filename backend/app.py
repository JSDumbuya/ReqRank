import json
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from transformers import pipeline
from bertopic import BERTopic
from collections import defaultdict
from preprocess import *
from dependencies import *
from classify_reqs import * 
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/prioritize")
async def prioritize_requirements(
    requirements: str = Form(...),
    normalizedWeights: str = Form(...),
    normalizedNfrWeights: str = Form(...),
    stakeholdersPrioritized: str = Form(...),
    stakeholderNames: List[str] = Form(...),
    stakeholderWeights: List[str] = Form(...),
    stakeholderFile: List[UploadFile] = File(...),
    includeEffort: str = Form(...),
    includeCost: str = Form(...)
):
    # Receive data from frontend
    requirements = json.loads(requirements)
    normalized_weights = json.loads(normalizedWeights)
    normalized_nfr_weights = json.loads(normalizedNfrWeights)
    is_stakeholders_prioritized = json.loads(stakeholdersPrioritized)
    includeEffort = json.loads(includeEffort)
    includeCost = json.loads(includeCost)

    '''requirements_file_path = ""
    if requirements:
        file_path = os.path.join(UPLOAD_DIR, "requirementsfile.txt")
        
        with open(file_path, mode="w", newline="") as file:
            for req in requirements:
                file.write(f"{req['text']}\n")
        
        requirements_file_path = file_path'''

    stakeholders = []

    for i in range(len(stakeholderNames)):
        stakeholder_data = {
            'name': stakeholderNames[i],
            'weight': int(stakeholderWeights[i]) if is_stakeholders_prioritized else 1, 
            'id': i
            }

        if stakeholderFile: 
            file = stakeholderFile[i]
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            stakeholder_data['file'] = file_path

        stakeholders.append(stakeholder_data)
    
    # Prioritize requirements

    # 1. Sentiment + topic extraction
    sentence_meta_data, all_sentences, all_embeddings, representation_type, vectorizer = prepare_stakeholders(stakeholders)
    topic_model, topic_sentiment_scores, topic_popularity_scores = derive_topics_and_score(sentence_meta_data, all_sentences, all_embeddings, vectorizer)
    
    # 2. Relatedness + prep of requirements
    prepared_requirements = prepare_requirements(requirements)

    # 3. Relate stakeholder feedback to reqs
    prepared_requirements = relate_topics_to_reqs_and_score(prepared_requirements, topic_model, topic_sentiment_scores, topic_popularity_scores, representation_type, vectorizer)

    # 4. Classification
    prepared_requirements = classify_requirements(prepared_requirements, normalized_nfr_weights)

    # 5. prioritize, rank, add explanations
    prioritized_requirements = calculate_final_score_and_rank(prepared_requirements, normalized_weights, includeCost, includeEffort)
    prioritized_requirements = generate_explanation(prioritized_requirements, includeCost, includeEffort)
    prioritized_requirements = convert_ndarrays(prioritized_requirements)

    # 6. Delete files in uploads if everythings is okay
    try:
        '''if requirements_file_path and os.path.exists(requirements_file_path):
            os.remove(requirements_file_path)'''

        for stakeholder in stakeholders:
            if 'file' in stakeholder and os.path.exists(stakeholder['file']):
                os.remove(stakeholder['file'])
    
    except Exception as e:
        print(f"Error deleting files: {e}")

    return JSONResponse(
        content={"message": "Prioritization completed!", "prioritized_data": prioritized_requirements},
        status_code=200
    )



# -------- utils ----------

def convert_ndarrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarrays(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    else:
        return obj
    
def generate_explanation(requirements, include_cost, include_effort):
    for req in requirements:
        exp_parts = []

        if req["type"] == "FR":
            exp_parts.append("Type: Functional Requirement – describes what the system should do.")
            exp_parts.append(
                f"Normalized NFR Importance Score: {req['normalized_nfr_importance_score']:.2f} – based on the average importance of the non-functional requirements linked to this group."
            )
        else:
            exp_parts.append(f"Type: Non-Functional Requirement ({req['category']}) – defines qualities or constraints of the system.")
            exp_parts.append(
                f"Normalized NFR Importance Score: {req['normalized_nfr_importance_score']:.2f} – based on the priority you assigned to its category."
            )

        exp_parts.append(
            f"Normalized Topic Score: {req['normalized_topic_score']:.2f} – reflects how many unique stakeholders discussed the topic related to this requirement. "
            "This helps highlight areas of shared interest or concern."
        )

        exp_parts.append(
            f"Normalized Sentiment Score: {req['normalized_sentiment_score']:.2f} – based on the average negative sentiment from stakeholder feedback in this topic. "
            "Negative feedback is weighted more heavily to prioritize issues that need attention."
        )

        exp_parts.append(
            f"Normalized Relatedness Score: {req['normalized_relatedness_score']:.2f} – based on the number of related requirements, which is {req['group_size']}."
        )

        if include_cost:
            exp_parts.append(
                f"Normalized Ressource Cost Score: {req['normalized_cost_score']:.2f} – represents the estimated cost impact of implementing this requirement."
            )

        if include_effort:
            exp_parts.append(
                f"Normalized Implementation Effort Score: {req['normalized_effort_score']:.2f} – represents the estimated development effort required for this requirement."
            )


        exp_parts.append(
            f"Final Score: {req['final_score']:.2f} – calculated as a weighted combination of the above factors based on your chosen prioritization rules."
        )

        req['explanation'] = "\n\n".join(exp_parts)

    return requirements

 

# -------- Stakeholders and feedback ----------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def prepare_stakeholders(list_of_stakeholders):
    all_sentences = []
    all_embeddings = []
    sentence_metadata = []

    for stakeholder in list_of_stakeholders:
        file_path = stakeholder["file"]
        weight = stakeholder["weight"]
        name = stakeholder["name"]
        id = stakeholder["id"]
        if file_path:
            preprocessed_sentences_sbert = preprocess_embeddings(file_path)
            preprocessed_sentences_sa = preprocess_sentiment_analysis(file_path)
            sentiment_results = sentiment_pipeline(preprocessed_sentences_sa)

            #print(len(preprocessed_sentences_sbert))

            #500 = somewhat arbitrary -> topic eval bertopic performed well on 304 sentences
            if len(preprocessed_sentences_sbert) >= 500:
                embeddings = embedding_model.encode(preprocessed_sentences_sa, show_progress_bar=False)
                representation_type = "bertopic"
            else:
                vectorizer = TfidfVectorizer()
                embeddings = vectorizer.fit_transform(preprocessed_sentences_sa)
                representation_type = "lda"
           
        
        #Sentence (closest to original) is created to keep track of sentences accross different preprocessing steps for different tasks e.g. topic vs. SA
        for i, sentence in enumerate(preprocessed_sentences_sbert):
            sentence_metadata.append({
                "sentence": sentence,
                "embedding": embeddings[i],
                "representation_type": representation_type,
                "sentiment": sentiment_results[i],
                "stakeholder_weight": weight,
                "stakeholder_name": name,
                "stakeholder_id": id
            })
            all_sentences.append(sentence)
            all_embeddings.append(embeddings[i])

    return sentence_metadata, all_sentences, all_embeddings, representation_type, vectorizer

def derive_topics_and_score(sentence_metadata, all_sentences, all_embeddings, vectorizer):
    representation_type = sentence_metadata[0]["representation_type"]
    topics = []

    if representation_type == "bertopic":
        all_embeddings = np.array(all_embeddings)
        topic_model = BERTopic()
        topics, _ = topic_model.fit_transform(all_sentences, all_embeddings)

    elif representation_type == "lda":
        doc_term_matrix = vectorizer.fit_transform(all_sentences)
        #Dynamically choosing n_components -> #5, 10, 20
        num_sentences = len(all_sentences)
        n_components = max(5, min(num_sentences // 25, 20))
        lda_model = LatentDirichletAllocation(n_components=n_components, random_state=42)
        doc_topics = lda_model.fit_transform(doc_term_matrix)
        topics = np.argmax(doc_topics, axis=1)
        topic_model = lda_model

    for i, topic in enumerate(topics):
        sentence_metadata[i]["topic"] = int(topic)
    
    topic_sentiments = defaultdict(list)
    topic_stakeholder_weights = defaultdict(list)

    for entry in sentence_metadata:
        sentiment = entry["sentiment"]
        label = sentiment["label"]
        score = sentiment["score"]
        weight = entry["stakeholder_weight"]
        topic = entry["topic"]
        stakeholder_id = entry["stakeholder_id"]

        if label == "NEGATIVE":
            weighted_score = score * weight
            topic_sentiments[topic].append(weighted_score)
        
        #For popularity, register stakeholder influence only once per topic
        topic_stakeholder_weights[topic].append(weight)
    
    topic_sentiment_scores = {}
    topic_popularity_scores = {}

    for topic in topic_stakeholder_weights:
        #Avg negative sentiment score
        scores = topic_sentiments[topic]
        avg_sentiment = sum(scores) / len(scores) if scores else 0
        topic_sentiment_scores[topic] = avg_sentiment

        #Popularity
        stakeholder_weights = topic_stakeholder_weights[topic]
        total_popularity = sum(stakeholder_weights)
        topic_popularity_scores[topic] = total_popularity
    
    return topic_model, topic_sentiment_scores, topic_popularity_scores

def relate_topics_to_reqs_and_score(requirements, topic_model, topic_sentiment_scores, topic_popularity_scores, representation_type, vectorizer):
    topic_representations = []
    topic_ids = []

    if representation_type == "bertopic":
        for topic in topic_model.get_topic_info().index:
            words = topic_model.get_topic(topic)
            topic_text = " ".join([word for word, _ in words]) if words else ""
            if topic_text:
                topic_representations.append(topic_text)
                topic_ids.append(topic)
        topic_embeddings = embedding_model.encode(topic_representations, show_progress_bar=False)

        for req in requirements.values():
            req_emb = req['embedding']
            similarities = cosine_similarity([req_emb], topic_embeddings)[0]
            best_topic_idx = int(np.argmax(similarities))
            best_topic_id = topic_ids[best_topic_idx]
            best_similarity = float(similarities[best_topic_idx])

            req['topic_id'] = best_topic_id
            req['topic_score'] = topic_popularity_scores.get(best_topic_id, 0.0) * best_similarity
            req['sentiment_score'] = topic_sentiment_scores.get(best_topic_id, 0.0) * best_similarity
            req['cosine_similarity_score'] = best_similarity

    elif representation_type == "lda":
        for topic in range(topic_model.n_components):
            topic_ids.append(topic)

        req_texts = [req["text"] for req in requirements.values()]
        vectorized_reqs = vectorizer.transform(req_texts)
        topic_distributions = topic_model.transform(vectorized_reqs)

        for i, req in enumerate(requirements.values()):
            topic_probs = topic_distributions[i]
            best_topic_idx = int(np.argmax(topic_probs))
            best_topic_probability = float(topic_probs[best_topic_idx]) 

            req['topic_id'] = best_topic_idx
            req['topic_score'] = topic_popularity_scores.get(best_topic_idx, 0.0) * best_topic_probability
            req['sentiment_score'] = topic_sentiment_scores.get(best_topic_idx, 0.0) * best_topic_probability
            req['topic_probability'] = best_topic_probability

    return requirements


# --------Requirements ----------

def prepare_requirements(recieved_reqs):
    requirements = {}
    '''content = read_file(file_path)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    req_df = pd.DataFrame(lines[1:], columns=["req_text"])''' 

    req_texts = [req["text"] for req in recieved_reqs]
    req_df = pd.DataFrame(req_texts, columns=["req_text"])
   
    preprocessed_requirements = preprocess_reqs(req_df)
    requirement_embeddings = embedding_model.encode(preprocessed_requirements, show_progress_bar=False)

    dependencies = identify_dependencies(requirement_embeddings)
    groups = group_dependencies(len(preprocessed_requirements), dependencies)

    related_map = defaultdict(list)
    for i, j, _ in dependencies:
        related_map[i].append(j)
        related_map[j].append(i)

    for i, req_text in enumerate(preprocessed_requirements):
        org_req = recieved_reqs[i]
        group = None
        group_size = 0
        group_nr = None 

        for idx, g in enumerate(groups, start=1):
            if i in g:
                group = g
                group_size = len(g)
                group_nr = idx
                break
        related_to = related_map[i]
        related_count = len(related_to)

        requirements[i] = {
            'id': i,
            'text': req_text,
            'embedding': requirement_embeddings[i],
            'related_count': related_count,
            'related_to': related_to,
            'group': group,
            'group_size': group_size,
            'group_nr': group_nr,
            'sentiment_score': 0.0,
            'topic_score': 0.0,
            'cost_score': org_req.get("cost", 0.0),
            'effort_score': org_req.get("effort", 0.0), 
            'nfr_importance_score': 0.0,
            'relatedness_score': np.log1p(group_size),
            'final_score': 0.0
        }
    
    return requirements

# -------- Classification ----------

def classify_requirements(prepared_requirements, nfrweights):
    #Binary class
    fr_ids, nfr_ids = binary_classification(prepared_requirements)

    for req_id in fr_ids:
        prepared_requirements[req_id]["type"] = "FR"
    for req_id in nfr_ids:
        prepared_requirements[req_id]["type"] = "NFR" 
    
    #Multiclass  
    category_ids = multi_class_classification(prepared_requirements)

    for category, req_ids in category_ids.items():
        for req_id in req_ids:
            prepared_requirements[req_id]["category"] = category
        
    #Scoring
    group_nfr_scores = defaultdict(list)
    #NFRs are given the weights as a score 
    for req in prepared_requirements.values():
        if req.get("type") == "NFR":
            category = req.get("category")
            if category in nfrweights:
                req["nfr_importance_score"] = nfrweights[category]
    
    for req in prepared_requirements.values():
        if req.get("type") == "NFR":
            group = req.get("group_nr")
            group_nfr_scores[group].append(req.get("nfr_importance_score", 0))
    #The average nfr score is given as a boost to the related frs
    group_avg_scores = {group: sum(scores) / len(scores) for group, scores in group_nfr_scores.items()}

    for req in prepared_requirements.values():
        if req.get("type") == "FR":
            group = req.get("group_nr")
            req["nfr_importance_score"] = group_avg_scores.get(group, 0)

    return prepared_requirements

def calculate_final_score_and_rank(requirements, normalized_weights, include_cost, include_effort):
    frontend_weight_to_score_map = {
        'sentiment': 'normalized_sentiment_score',
        'popularity': 'normalized_topic_score',
        'nfrImportance': 'normalized_nfr_importance_score',
        'amountRelatedReqs': 'normalized_relatedness_score',
    }

    if include_cost:
        frontend_weight_to_score_map['cost'] = 'normalized_cost_score'
    if include_effort:
        frontend_weight_to_score_map['effort'] = 'normalized_effort_score'

    # Normalize scores for all requirements
    normalized_requirements = normalize_scores(requirements, include_cost, include_effort)

    # Calculate the final score for each requirement
    '''for req in normalized_requirements.values():
        req['final_score'] = sum(req[frontend_weight_to_score_map[key]] * normalized_weights[key] 
                                 for key in frontend_weight_to_score_map if key in normalized_weights)'''
    
    for req in normalized_requirements.values():
        final_score = 0.0

        for key, score_key in frontend_weight_to_score_map.items():
            weight = normalized_weights[key]
            score = req[score_key]
            #Subract
            if key in ['cost', 'effort']:
                final_score -= score * weight
            #Add
            else:
                final_score += score * weight

        req['final_score'] = final_score
    
    # Group the requirements by their group number
    group_scores = defaultdict(list)
    for req in normalized_requirements.values():
        group_scores[req['group_nr']].append(req['final_score'])

    # Calculate the average score for each group
    group_avg_scores = {group_nr: np.mean(scores) for group_nr, scores in group_scores.items()}

    # Sort the groups by their average score, highest to lowest
    sorted_groups = sorted(group_avg_scores.items(), key=lambda x: x[1], reverse=True)
    group_ranks = {group_nr: rank + 1 for rank, (group_nr, _) in enumerate(sorted_groups)}

    # Group the requirements by their group number for sorting within each group
    grouped_reqs = defaultdict(list)
    for req in normalized_requirements.values():
        grouped_reqs[req['group_nr']].append(req)
    
    # Sort the requirements within each group by their final score
    for group_nr, group_reqs in grouped_reqs.items():
        group_reqs.sort(key=lambda r: r['final_score'], reverse=True)
        
        # Assign ranks within the group
        for i, req in enumerate(group_reqs):
            req['rank'] = i + 1
            req['group_avg_score'] = group_avg_scores[group_nr]
            req['group_rank'] = group_ranks[group_nr]

    # Flatten the grouped requirements and sort all requirements globally by final score
    sorted_reqs = [req for group_reqs in grouped_reqs.values() for req in group_reqs]
    sorted_reqs.sort(key=lambda r: (r['group_rank'], -r['final_score']))

    return sorted_reqs


def normalize_scores(requirements, include_cost, include_effort):
    score_keys = ['sentiment_score', 'topic_score', 'nfr_importance_score', 'relatedness_score']
    if include_cost:
        score_keys.append('cost_score')
    if include_effort:
        score_keys.append('effort_score')
    
    scores = np.array([[req[key] for key in score_keys] for req in requirements.values()])

    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores)

    for i, req in enumerate(requirements.values()):
        for j, score_key in enumerate(score_keys):
            req[f'normalized_{score_key}'] = normalized_scores[i][j]
    
    return requirements

#To run
# uvicorn app:app --reload
#Host: 8000

    


