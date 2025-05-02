import json
import pickle
from sentence_transformers import SentenceTransformer

# Load parenting Q&A dataset
with open("data.json", "r") as f:
    data = json.load(f)

qa_pairs = []
for topic in data["toddler_care"]:
    qa_pairs.extend(data["toddler_care"][topic])

questions = [item["question"] for item in qa_pairs]

# Load lightweight embedding model
model = SentenceTransformer("paraphrase-albert-small-v2")

# Generate embeddings
embeddings = model.encode(questions, convert_to_tensor=True)

# Save to app/embeddings.pkl
with open("app/embeddings.pkl", "wb") as f:
    pickle.dump({
        "questions": questions,
        "answers": qa_pairs,
        "embeddings": embeddings
    }, f)

print("✅ embeddings.pkl created successfully in /app")
