import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Global variables
model = None
index = None
qa_pairs = []

def load_qa_data():
    global qa_pairs
    if qa_pairs:
        return
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
        vectors = model.encode(questions, convert_to_numpy=True)
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

def search_answer(query, top_k=1):
    load_qa_data()
    load_model_and_index()

    qvec = model.encode([query], convert_to_numpy=True)
    qvec = qvec / np.linalg.norm(qvec)
    _, idx = index.search(qvec, top_k)

    result = [qa_pairs[i]["answer"] for i in idx[0]]
    return result[0]
