import pandas as pd
import numpy as np
import pickle
from utils import read_csv
from preprocess import preprocess_reqs
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import and clean data
full_NFR_data = read_csv("backend/datasets/NFR_PROMISE-kopi.csv")
requirementtext_nfr_file_path = "backend/datasets/requirementtext_NFR_PROMISE-kopi.csv"

cleaned_nfrs = preprocess_reqs(requirementtext_nfr_file_path)
cleaned_nfrs_df = pd.DataFrame({'cleaned_nfr': cleaned_nfrs})

full_NFR_data['cleaned_nfr'] = cleaned_nfrs_df['cleaned_nfr']
X = full_NFR_data['cleaned_nfr']
y = full_NFR_data['class']

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = model.encode(X.tolist(), show_progress_bar=True)

# Dataset split + oversample
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_embeddings, y)
print("Original distribution:", Counter(y))
print("After SMOTE:", Counter(y_resampled))
#X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Tune C + gamma
param_grid = {
    'C': [0.1, 1.0, 10.0], 
    'gamma': ['scale', 0.1, 0.01], 
    'class_weight': ['balanced'] 
}

#cv = cross validation.
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

# Create model
#print("Best Parameters:", grid_search.best_params_)
best_rbf_model = grid_search.best_estimator_

y_pred = best_rbf_model.predict(X_test) 

# Evalution
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("rbf_model_multi.pkl", "wb") as model_file:
    pickle.dump(best_rbf_model, model_file)