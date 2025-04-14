import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_reqs
from utils import read_csv
import pickle
import numpy as np


full_data = read_csv("/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/Promise_exp/PROMISE_exp.csv")
requirement_text_file_path = "/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/Promise_exp/requirement_text_PROMISE.csv"

cleaned_reqs = preprocess_reqs(requirement_text_file_path)
cleaned_reqs_df = pd.DataFrame({'cleaned_reqs': cleaned_reqs})

full_data['binary_class'] = full_data['class'].apply(lambda x: 'FR' if x == 'F' else 'NFR')

# Generate features
# First try - multi-class: accuracy = 0,67
# vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(cleaned_reqs_df['cleaned_reqs'])
y = full_data['binary_class']

# Split into training + test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
svm_model_binary = SVC(kernel='linear', C=1.0, class_weight='balanced')
svm_model_binary.fit(X_train, y_train)

# Predict
y_pred = svm_model_binary.predict(X_test)

# Evaluate - gider vi at beholde 10-fold validation score?
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(svm_model_binary, X, y, cv=10, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average 10-fold CV accuracy: {np.mean(cv_scores)}")

# Save models
with open("svm_model_binary.pkl", "wb") as model_file:
    pickle.dump(svm_model_binary, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
