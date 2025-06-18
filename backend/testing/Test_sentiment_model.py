import pandas as pd
from transformers import pipeline
from preprocess import preprocess_lists
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
df = pd.read_csv('backend/datasets/filtered_sentiment_sampled.csv')

texts = df['content'].tolist()
labels = df['polarity'].tolist()

preprocessed_texts = preprocess_lists(texts)

results = [(data, sentiment_pipeline(data)[0]) for data in preprocessed_texts]

pred_labels = []
for text in preprocessed_texts:
    result = sentiment_pipeline(text)[0] 
    pred_label = 1 if result['label'] == 'POSITIVE' else 0
    pred_labels.append(pred_label)

precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='binary')
accuracy = accuracy_score(labels, pred_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


#File creation
''''
input_path = 'backend/datasets/sentiment-analysis-dataset-google-play-app-reviews.csv'
df = pd.read_csv(input_path)

df = df[df['score'] != 3][['content', 'score']]


def map_polarity(score):
    if score in [1, 2]:
        return 0  # negative
    elif score in [4, 5]:
        return 1  # positive
    else:
        return None

df['polarity'] = df['score'].apply(map_polarity)


df = df[['content', 'polarity']]


negative_samples = df[df['polarity'] == 0].sample(n=500, random_state=42)
positive_samples = df[df['polarity'] == 1].sample(n=500, random_state=42)


sampled_df = pd.concat([negative_samples, positive_samples]).reset_index(drop=True)


output_path = 'backend/datasets/filtered_sentiment_sampled.csv'
sampled_df.to_csv(output_path, index=False)'''


