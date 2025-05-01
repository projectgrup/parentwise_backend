import json
import pickle
from sentence_transformers import SentenceTransformer

# Load your parenting Q&A dataset
with open("app/data.json", "r") as f:
    data = json.load(f)

# Flatten questions
qa_pairs = []
for topic in data.get("toddler_care", {}):
    qa_pairs.extend(data["toddler_care"][topic])

questions = [item["question"] for item in qa_pairs]

# Load lightweight model
model = SentenceTransformer("paraphrase-albert-small-v2")

# Generate embeddings
embeddings = model.encode(questions, convert_to_tensor=True)

# Save to embeddings.pkl
with open("app/embeddings.pkl", "wb") as f:
    pickle.dump({
        "questions": questions,
        "answers": qa_pairs,
        "embeddings": embeddings
    }, f)

print("✅ embeddings.pkl created successfully!")
