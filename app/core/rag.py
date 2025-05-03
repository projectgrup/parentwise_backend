import redis
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

model = None
index = None
qa_pairs = []

def load_qa_data():
    global qa_pairs
    try:
        with open("app/data.json", "r") as f:
            data = json.load(f)
            for topic in data.get("toddler_care", {}):
                qa_pairs.extend(data["toddler_care"][topic])
        print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
    except Exception as e:
        print("❌ Failed to load Q&A data:", e)

def load_model_and_index():
    global model, index
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if index is None:
        questions = [pair["question"] for pair in qa_pairs]
        vectors = model.encode(questions)
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

def search_answer(query, top_k=1):
    cached = redis_client.get(query)
    if cached:
        return eval(cached)[0]

    load_model_and_index()
    qvec = model.encode([query])
    qvec = qvec / np.linalg.norm(qvec)
    _, idx = index.search(qvec, top_k)

    result = [qa_pairs[i]["answer"] for i in idx[0]]
    redis_client.set(query, str(result), ex=86400)
    return result[0]
