from dependencies import identify_dependencies, group_dependencies
from embeddings_topic import generate_embeddings, embedding_model
from utils import read_csv
import pandas as pd


full_set = read_csv("backend/dependency_set.csv")
text_dep_file_path = "backend/text_dependency_set.csv"
sentences, dep_embeddings = generate_embeddings(text_dep_file_path, embedding_model)
#Foretag yderligere split af preprocessing pipeline, fordi generate embeddings bruger preprocess general - so split this
deps = identify_dependencies(dep_embeddings, 0.7)
group_dependencies(len(deps), deps)

df = pd.DataFrame(group_dependencies)
df.to_csv('test_dep.csv', index=False)