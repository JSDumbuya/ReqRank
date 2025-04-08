
def score_requirement(requirement):
    return 0
    

def sort_requirements(requirement_scores):
    sorted_priorities = sorted(requirement_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_priorities