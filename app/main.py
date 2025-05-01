from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, util
import os, json

app = FastAPI()
model = SentenceTransformer("paraphrase-albert-small-v2")

# Load data.json safely
try:
    file_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(file_path, "r") as f:
        data = json.load(f)
except Exception as e:
    print("❌ data.json missing:", str(e))
    data = {"toddler_care": {}}

qa_pairs = []
for topic in data["toddler_care"]:
    qa_pairs.extend(data["toddler_care"][topic])

questions = [item["question"] for item in qa_pairs]
question_embeddings = model.encode(questions, convert_to_tensor=True)

@app.post("/ask_question")
async def ask_question(req: Request):
    body = await req.json()
    question = body["question"]

    query_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    answer = qa_pairs[scores.argmax().item()]["answer"]

    return {"answer": answer}

@app.post("/generate_schedule")
async def generate_schedule(req: Request):
    body = await req.json()
    age = body["age"]
    wake = body["wake_time"]
    nap = body["nap_pref"]
    meals = body["meals"]

    lines = [
        f"Wake up at {wake}, then breakfast.",
        "Morning activity or play time.",
        "Mid-morning nap." if nap == "2 naps" else "",
        "Lunch and story time.",
        "Afternoon nap." if nap in ["1 nap", "2 naps"] else "",
        "Outdoor play.",
        "Dinner and wind-down activities.",
        "Bedtime around 8 PM."
    ]
    routine = "\n".join([line for line in lines if line])
    return {"routine": routine}

@app.post("/submit_feedback")
async def submit_feedback(req: Request):
    body = await req.json()
    with open("feedback_log.json", "a") as f:
        json.dump(body, f)
        f.write("\n")
    return {"status": "success"}
