import pandas as pd
import numpy as np
import pickle
from utils import read_csv
from preprocess import preprocess_reqs
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import and clean data

full_NFR_data = read_csv("/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/Promise_exp/NFR_PROMISE.csv")
requirementtext_nfr_file_path = "/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/Promise_exp/requirementtext_NFR_PROMISE.csv"

cleaned_nfrs = preprocess_reqs(requirementtext_nfr_file_path)
cleaned_nfrs_df = pd.DataFrame({'cleaned_nfr': cleaned_nfrs})

full_NFR_data['cleaned_nfr'] = cleaned_nfrs_df['cleaned_nfr']
X = full_NFR_data['cleaned_nfr']
y = full_NFR_data['class']

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = model.encode(X.tolist(), show_progress_bar=True)

# Dataset split
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, stratify=y, random_state=42)


# Tune C + gamma
param_grid = {
    'C': [0.1, 1.0, 10.0], 
    'gamma': ['scale', 0.1, 0.01], 
    'class_weight': ['balanced'] 
}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

# Create model
#print("Best Parameters:", grid_search.best_params_)
best_rbf_model = grid_search.best_estimator_

y_pred = best_rbf_model.predict(X_test) 

# Evalution

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(best_rbf_model, X_embeddings, y, cv=10, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average 10-fold CV accuracy: {np.mean(cv_scores):.4f}")

# Save model
with open("rbf_model_multi.pkl", "wb") as model_file:
    pickle.dump(best_rbf_model, model_file)