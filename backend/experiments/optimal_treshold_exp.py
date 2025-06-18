from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from preprocess import preprocess_reqs, preprocess_reqs_clustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

def load_data(source='voyager'):
    if source == 'estore':
        df = pd.read_csv('backend/datasets/E_Store_set.csv')
        reqs = df[['requirement_text']].dropna()
        labels = df['group_id'].dropna().astype(int).tolist()
    else:
        df = pd.read_csv('backend/datasets/Voyager_reqs_treshold.csv')
        reqs = df[['requirementtext']].dropna()
        labels = df['group_id'].dropna().astype(int).tolist()
    return reqs, labels


def evaluate_ahc_clustering_on_embeddings(reqs_df, true_labels, thresholds=np.arange(0.50, 0.86, 0.05)):
    preprocessed_requirements = preprocess_reqs_clustering(reqs_df) 

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(preprocessed_requirements, show_progress_bar=False)

    results = []

    for threshold in thresholds:
        dist_matrix = cosine_distances(embeddings)

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=1 - threshold
        )
        cluster_labels = clusterer.fit_predict(dist_matrix)

        ari = adjusted_rand_score(true_labels, cluster_labels)
        if 2 <= len(set(cluster_labels)) < len(embeddings):
            sil = silhouette_score(embeddings, cluster_labels, metric='cosine')
        else:
            sil = np.nan

        results.append({
            'Threshold': round(threshold, 2),
            'ARI': ari,
            'Silhouette': sil,
            'NumClusters': len(set(cluster_labels))
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    reqs_df, true_labels = load_data(source='voyager')
    results_df = evaluate_ahc_clustering_on_embeddings(reqs_df, true_labels)
    print(results_df)

    results_df.set_index('Threshold')[['ARI', 'Silhouette']].plot(
        marker='o', figsize=(10, 6), title="Agglomerative Clustering on Requirement Embeddings"
    )
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''
Manual inspection - old method cosine + dfs

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

