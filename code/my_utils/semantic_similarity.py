from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def compute_cosine_similarity(embeddings1, embeddings2):
    cosine_sim = [cosine_similarity(embeddings1.reshape(1, -1), embeddings2[i,:].reshape(1, -1))[0] for i in range(embeddings2.shape[0])]
    return np.mean(cosine_sim)

def extract_semantic_embeddings(sentence):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model.encode(sentence)

def __main__():
    a = extract_semantic_embeddings("sentence").reshape(1, -1)
    b = extract_semantic_embeddings("phrase").reshape(1, -1)

    c = compute_cosine_similarity(a, b)
    print(c)

__main__()