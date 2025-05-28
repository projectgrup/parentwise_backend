import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

model = None
index = None
qa_pairs = []

def init_rag():
    global model, index, qa_pairs
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    if not qa_pairs:
        try:
            with open("app/data.json", "r") as f:
                data = json.load(f)
                for topic in data.get("toddler_care", {}):
                    qa_pairs.extend(data["toddler_care"][topic])
            print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
        except Exception as e:
            print(f"❌ Failed to load Q&A data: {e}")

    if index is None:
        questions = [pair["question"] for pair in qa_pairs]
        vectors = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        print("✅ FAISS index built.")

def search_answer(query, top_k=1):
    global model, index, qa_pairs
    if not model or not index or not qa_pairs:
        init_rag()

    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    _, idx = index.search(qvec, top_k)

    result = [qa_pairs[i]["answer"] for i in idx[0] if i < len(qa_pairs)]
    return result[0] if result else "Sorry, no relevant answer found."
