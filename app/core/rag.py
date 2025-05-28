import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Globals
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2')
index = None
qa_pairs = []

def load_qa_data():
    global qa_pairs
    if qa_pairs:
        return
    try:
        with open("app/data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for topic in data.get("toddler_care", {}):
                qa_pairs.extend(data["toddler_care"][topic])
        print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
    except Exception as e:
        print("❌ Q&A Load Error:", e)

def build_index():
    global index
    if index is not None:
        return
    questions = [pair["question"] for pair in qa_pairs]
    vectors = model.encode(questions, convert_to_numpy=True)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"✅ Built FAISS index with {len(questions)} entries.")

def search_answer(query, top_k=1):
    if not qa_pairs:
        load_qa_data()
    if index is None:
        build_index()

    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec)
    _, idx = index.search(query_vec, top_k)
    return qa_pairs[idx[0][0]]["answer"]
