import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from preprocess import preprocess_classification_experiments
from utils import read_csv
import pickle

#Trained on the promise_exp dataset.
full_data = read_csv("")
requirement_text_file_path = ""

cleaned_reqs = preprocess_classification_experiments(requirement_text_file_path)
cleaned_reqs_df = pd.DataFrame({'cleaned_reqs': cleaned_reqs})

full_data['binary_class'] = full_data['class'].apply(lambda x: 'FR' if x == 'F' else 'NFR')

# Generate features
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

# Evaluate
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save models
with open("svm_model_binary.pkl", "wb") as model_file:
    pickle.dump(svm_model_binary, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
