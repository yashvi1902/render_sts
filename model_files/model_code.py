from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_name ='nli-distilroberta-base-v2'

def get_embeddings(s1):
    model = SentenceTransformer(model_name)
    se = model.encode(s1)
    return se

def get_similarity_score(s1,s2):
    return cosine_similarity(s1,s2)

