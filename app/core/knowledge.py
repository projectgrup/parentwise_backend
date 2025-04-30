import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# Define dataset path
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../../dataset.json")

# Load your static dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)["toddler_care"]

flat_qna = []
for category in raw_data:
    for item in raw_data[category]:
        flat_qna.append((item["question"], item["answer"]))

questions = [q for q, _ in flat_qna]
answers = [a for _, a in flat_qna]

# Embed all questions
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
q_embeddings = model.encode(questions)
q_embeddings = q_embeddings / np.linalg.norm(q_embeddings, axis=1)[:, None]

# Create FAISS index
index = faiss.IndexFlatIP(q_embeddings.shape[1])
index.add(q_embeddings)

def search_answer(query: str, top_k: int = 1):
    q_vec = model.encode([query])
    q_vec = q_vec / np.linalg.norm(q_vec)
    _, idx = index.search(q_vec, top_k)
    return answers[idx[0][0]]
