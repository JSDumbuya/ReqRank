
import pandas as pd

full_data = pd.read_csv("/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/FR_NFR_Dataset/FR_NFR_Dataset.csv")

full_data[['Requirement Text']].to_csv("/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/4. semester/Thesis/Datasets/FR_NFR_Dataset/FR_NFR_Dataset_requirementtext.csv", index=False)
