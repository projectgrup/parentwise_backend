from fastapi import FastAPI, Request
import json, os, pickle
import torch
from sentence_transformers import SentenceTransformer, util
from random import choice

app = FastAPI()

# ✅ Load precomputed embeddings
try:
    file_path = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
    with open(file_path, "rb") as f:
        emb_data = pickle.load(f)
        questions = emb_data["questions"]
        qa_pairs = emb_data["answers"]
        question_embeddings = emb_data["embeddings"]
except Exception as e:
    print("❌ embeddings.pkl load error:", str(e))
    questions, qa_pairs, question_embeddings = [], [], None

# ✅ Q&A route using semantic search
@app.post("/ask_question")
async def ask_question(req: Request):
    try:
        body = await req.json()
        question = body.get("question", "").strip()
        if not question or question_embeddings is None:
            return {"answer": "Q&A system not ready or invalid input."}

        model = SentenceTransformer("paraphrase-albert-small-v2")
        query_embedding = model.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, question_embeddings)[0]
        best_idx = torch.argmax(scores).item()
        return {"answer": qa_pairs[best_idx]["answer"]}
    except Exception as e:
        print("❌ /ask_question error:", e)
        return {"answer": "Something went wrong. Try again."}

# ✅ Toddler schedule generator
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

# ✅ Feedback logging
@app.post("/submit_feedback")
async def submit_feedback(req: Request):
    body = await req.json()
    with open("feedback_log.json", "a") as f:
        json.dump(body, f)
        f.write("\n")
    return {"status": "success"}

# ✅ Simple story generator
@app.post("/story/generate")
async def generate_story(req: Request):
    body = await req.json()
    age = body.get("age", 3)
    theme = body.get("theme", "friendship").lower()

    stories = {
        "jungle": [
            f"Once upon a time, a curious {age}-year-old monkey explored the jungle and made new animal friends.",
            f"In a lush jungle, a baby elephant went on an adventure to find the tallest tree. What a journey!"
        ],
        "friendship": [
            f"A little rabbit learned how to share carrots with a new friend and became the happiest bunny in the meadow.",
            f"A {age}-year-old bear invited all the forest animals to a tea party. Everyone laughed and played together."
        ],
        "default": [
            f"Once there was a magical cloud that danced in the sky just for a {age}-year-old child.",
            f"A star fell to Earth and whispered bedtime wishes to every sleeping child, including you."
        ]
    }

    selected = stories.get(theme, stories["default"])
    return {"story": choice(selected)}

# ✅ Mock Firebase token verification
@app.post("/auth/verify")
async def verify_token(req: Request):
    body = await req.json()
    token = body.get("token", "").strip()
    if token == "demo-token":
        return {"status": "success", "user": "test_user"}
    else:
        return {"status": "error", "message": "Invalid token"}
