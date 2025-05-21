from dependencies import identify_dependencies, group_dependencies
from embeddings_topic import generate_embeddings, embedding_model
from utils import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from preprocess import preprocess_reqs

# Load and preprocess data
full_set = read_csv("backend/dependency_set.csv")
text_dep_file_path = "backend/text_dependency_set.csv"
normalized_reqs = preprocess_reqs(text_dep_file_path)

# Create embeddings
sentences, dep_embeddings = generate_embeddings(normalized_reqs, embedding_model)


'''
Experiment - automated with silhouette score, bss, wss
thresholds = np.arange(0.3, 0.7, 0.1)
silhouette_scores = []
bss_scores = []
wss_scores = []

def compute_bss_wss(embeddings, labels):
    wss = 0
    bss = 0
    overall_mean = np.mean(embeddings, axis=0)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        wss += np.sum((cluster_points - cluster_mean) ** 2)
        bss += len(cluster_points) * np.sum((cluster_mean - overall_mean) ** 2)

    return bss, wss

for threshold in thresholds:
    deps = identify_dependencies(dep_embeddings, threshold)

    # for dep in deps:
    #     print("This is a dependency group: ", dep[0], "-",  normalized_reqs[dep[0]], ",", dep[1], "-", normalized_reqs[dep[1]],",", "similarity score: ", dep[2])

    grouped_deps = group_dependencies(num_reqs=len(normalized_reqs), dependencies=deps)

    labels = np.full(len(normalized_reqs), -1)
    for cluster_id, group in enumerate(grouped_deps):
        for idx in group:
            labels[idx] = cluster_id

    clustered_mask = labels != -1
    filtered_embeddings = dep_embeddings[clustered_mask]
    filtered_labels = labels[clustered_mask]

    n_clusters = len(np.unique(filtered_labels))
    n_samples = len(filtered_labels)

    if 2 <= n_clusters < n_samples:
        silhouette = silhouette_score(filtered_embeddings, filtered_labels)
    else:
        silhouette = np.nan  # Silhouette score undefined in this case

    bss, wss = compute_bss_wss(filtered_embeddings, filtered_labels)

    silhouette_scores.append(silhouette)
    bss_scores.append(bss)
    wss_scores.append(wss)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(thresholds, silhouette_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Threshold")
plt.ylabel("Score")

plt.subplot(1, 3, 2)
plt.plot(thresholds, bss_scores, marker='o', color='green')
plt.title("Between-cluster Sum of Squares (BSS)")
plt.xlabel("Threshold")
plt.ylabel("Score")

plt.subplot(1, 3, 3)
plt.plot(thresholds, wss_scores, marker='o', color='red')
plt.title("Within-cluster Sum of Squares (WSS)")
plt.xlabel("Threshold")
plt.ylabel("Score")

plt.tight_layout()
plt.show()''' 



'''
Manual inspection

deps = identify_dependencies(dep_embeddings, 0.5)

for dep in deps:
    print("This is a dependency group: ", dep[0], "-",  normalized_reqs[dep[0]], ",", dep[1], "-", normalized_reqs[dep[1]],",", "similarity score: ", dep[2])

grouped_deps = group_dependencies(num_reqs=len(normalized_reqs), dependencies=deps)

grouped_text = [
    [normalized_reqs[i] for i in group]
    for group in grouped_deps
]

df = pd.DataFrame({
    "Group ID": list(range(len(grouped_text))),
    "Requirements": grouped_text
})

df.to_csv('test_dep.csv', index=False)'''

