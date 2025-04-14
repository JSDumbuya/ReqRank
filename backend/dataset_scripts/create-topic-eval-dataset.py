from utils import read_csv, create_csv

file_path = "/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/Google Play Store Apps/googleplaystore_user_reviews.csv"

df = read_csv(file_path)

filtered_df = df[df["App"] == "Firefox Browser fast & private"][["Translated_Review"]]

create_csv(filtered_df, "topic-Eval-Dataset.csv")


