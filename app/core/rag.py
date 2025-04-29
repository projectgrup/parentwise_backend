import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# ✅ Global references (lazy)
sample_data = ["What should I feed my toddler?", "How much sleep does a 2-year-old need?"]
index = None
model = None

def load_model_and_index():
    global model, index
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if index is None:
        vectors = model.encode(sample_data)
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

def search_question(query, top_k=3):
    cached = redis_client.get(query)
    if cached:
        return eval(cached)

    load_model_and_index()
    qvec = model.encode([query])
    qvec = qvec / np.linalg.norm(qvec)
    _, idx = index.search(qvec, top_k)
    result = [sample_data[i] for i in idx[0]]
    redis_client.set(query, str(result), ex=86400)
    return result
