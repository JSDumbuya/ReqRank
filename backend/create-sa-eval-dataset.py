from utils import *
import pandas as pd


def clean_data(dataset):
    rows_before_cleaning = dataset.shape[0]
    dataset_cleaned = dataset.dropna(subset=['Sentiment', 'Sentiment_Polarity'])
    rows_after_cleaning = dataset_cleaned.shape[0]
    
    print(f"Number of reviews removed: {rows_before_cleaning - rows_after_cleaning}")

    return dataset_cleaned

def select_balanced_samples(dataset, sample_size):
    positive_reviews = dataset[dataset['Sentiment'] == 'Positive']
    negative_reviews = dataset[dataset['Sentiment'] == 'Negative']

    num_samples = min(len(positive_reviews), len(negative_reviews), sample_size)

    positive_sample = positive_reviews.sample(n=num_samples, random_state=42)
    negative_sample = negative_reviews.sample(n=num_samples, random_state=42)

    balanced_dataset = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=42)

    print(f"Total data points: {balanced_dataset.shape[0]} (Positive: {num_samples}, Negative: {num_samples})")
    return balanced_dataset


google_play_store_dataset = read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/Google Play Store Apps/googleplaystore_user_reviews.csv')

cleaned_data = clean_data(google_play_store_dataset)
balanced_set = select_balanced_samples(cleaned_data, 500)
create_csv(balanced_set, "SA-Eval-Dataset")


