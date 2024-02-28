from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSimilarityCalculator:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
    
    def extract_semantic_embeddings(self, sentence):
        return self.model.encode(sentence)
    
    def compute_cosine_similarity(self, embeddings1, embeddings2):
        cosine_sim = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))
        return np.mean(cosine_sim)