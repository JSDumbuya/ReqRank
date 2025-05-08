
def score_requirement(requirement):
    return 0
    

def sort_requirements(requirement_scores):
    sorted_priorities = sorted(requirement_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_priorities


'''
priority_score = 
    w1 * sentiment_score * stakeholder_weight +
    w2 * topic_frequency +
    w3 * group_size +
    w4 * (1 - implementation_cost_normalized) +
    w5 * dependency_centrality_score -
    w6 * uncertainty_score
'''