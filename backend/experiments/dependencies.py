from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import spacy

nlp = spacy.load("en_core_web_sm")


'''
def extract_components(requirements):
    extracted = []
    
    for req in requirements:
        doc = nlp(req)
        components = set()
        
        for token in doc:
            if token.dep_ in {"dobj", "pobj", "nsubj", "nsubjpass"}:
                phrase = ""

                for child in token.children:
                    if child.dep_ in {"compound", "amod"}:
                        phrase += child.text + " "
                
                phrase += token.text
                components.add(phrase.strip().lower())
            
            if token.dep_ == "conj" and token.head.dep_ in {"dobj", "pobj"}:
                phrase = ""
                for child in token.children:
                    if child.dep_ in {"compound", "amod"}:
                        phrase += child.text + " "
                phrase += token.text
                components.add(phrase.strip().lower())
        
        extracted.append((req, components))
    
    return extracted
'''

def create_component_sentences(extracted_components):
    return [(req, " ".join(sorted(comp_set)) if comp_set else "empty") for req, comp_set in extracted_components]

def identify_dependencies(reqs_with_components, req_embeddings, component_embeddings, threshold=0.65):
    dependencies = []

    for i in range(len(component_embeddings)):
        for j in range(i + 1, len(component_embeddings)):
            similarity = cosine_similarity([component_embeddings[i]], [component_embeddings[j]])[0][0]

            if similarity >= threshold:
                dependencies.append({
                    "index_pair": (i, j),
                    "similarity": similarity,
                    "req_pair": (reqs_with_components[i][0], reqs_with_components[j][0]),
                    "req_embedding_pair": (req_embeddings[i], req_embeddings[j]),
                    "component_pair": (reqs_with_components[i][1], reqs_with_components[j][1])
                })

    return dependencies

'''def identify_dependencies(embeddings, threshold=0.65):
    dependencies = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]

            if similarity >= threshold:
                dependencies.append((i, j, similarity))
    
    return dependencies'''

def group_dependencies(num_reqs, dependencies):
    graph = defaultdict(list)
    #for i, j, _ in dependencies:
    for dep in dependencies:
        i, j = dep["index_pair"]
        graph[i].append(j)
        graph[j].append(i)
    
    visited = set()
    groups = []

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for i in range(num_reqs):
        if i not in visited:
            group = []
            dfs(i, group)
            if group:
                groups.append(group)

    return groups
