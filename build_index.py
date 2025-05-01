from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import pickle
import os

DATA_PATH = "app/core/data.json"
INDEX_PATH = "app/core/faiss.index"
QADATA_PATH = "app/core/qa_data.pkl"

# Load your Q&A data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)["toddler_care"]

flat_qna = []
for category in raw_data:
    for item in raw_data[category]:
        flat_qna.append((item["question"], item["answer"]))

questions = [q for q, _ in flat_qna]
answers = [a for _, a in flat_qna]

# Embed questions
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(questions)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Save FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

# Save answers
with open(QADATA_PATH, "wb") as f:
    pickle.dump(answers, f)

print("✅ FAISS index and answers saved!")
