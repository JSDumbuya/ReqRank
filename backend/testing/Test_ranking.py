import pandas as pd
from scipy.stats import kendalltau

#Load data
participant_path = "backend/datasets/user_ranked_outlook_reviews.csv"
reqrank_path = "backend/datasets/ReqRank_not_related_ranked.csv"
reqrank_relation_path = "backend/datasets/ReqRank_related.csv"

participant_df = pd.read_csv(participant_path, header=None, usecols=[1], names=["text_en"])
reqrank_df = pd.read_csv(reqrank_relation_path, header=None, names=["text"])

#Compare rankings
participant_rank_map = {req: rank for rank, req in enumerate(participant_df["text_en"], start=1)}

participant_ranks = []
reqrank_ranks = []

for req in participant_df["text_en"]:
    participant_ranks.append(participant_rank_map[req])
    if req in reqrank_df["text"].values:
        reqrank_rank = reqrank_df.index[reqrank_df["text"] == req].tolist()[0] + 1
    else:
        reqrank_rank = len(reqrank_df) + 1
    reqrank_ranks.append(reqrank_rank)

tau, p_value = kendalltau(participant_ranks, reqrank_ranks)

print(f"Kendall's Tau: {tau}")
print(f"P-value: {p_value}")