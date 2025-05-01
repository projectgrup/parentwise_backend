from fastapi import FastAPI, Request
import json, os

app = FastAPI()

# Load data.json
try:
    file_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(file_path, "r") as f:
        data = json.load(f)
except Exception as e:
    print("❌ data.json missing:", str(e))
    data = {"toddler_care": {}}

# Flatten Q&A pairs
qa_pairs = []
for category in data.get("toddler_care", {}):
    qa_pairs.extend(data["toddler_care"][category])

# Very simple keyword search (no ML)
def find_best_answer(query):
    query = query.lower()
    for pair in qa_pairs:
        if any(word in pair["question"].lower() for word in query.split()):
            return pair["answer"]
    return "Sorry, I couldn't find a matching answer."

@app.post("/ask_question")
async def ask_question(req: Request):
    body = await req.json()
    question = body.get("question", "")
    answer = find_best_answer(question)
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
