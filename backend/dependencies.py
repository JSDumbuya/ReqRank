from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def identify_dependencies(embeddings, threshold=0.6):
    dependencies = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]

            if similarity >= threshold:
                dependencies.append((i, j, similarity))
    
    return dependencies

def group_dependencies(num_reqs, dependencies):
    graph = defaultdict(list)
    for i, j, _ in dependencies:
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
