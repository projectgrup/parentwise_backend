from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import firebase_admin
from firebase_admin import credentials, auth
import json, os
from random import choice

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Firebase Admin
cred = credentials.Certificate("config/firebase_adminsdk.json")
firebase_admin.initialize_app(cred)

@app.get("/")
def read_root():
    return {"message": "✅ ParentWise backend is running."}

# Load data.json
try:
    with open("data.json", "r") as f:
        data = json.load(f)
    qa_pairs = []
    for topic in data.get("toddler_care", {}):
        qa_pairs.extend(data["toddler_care"][topic])
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
except Exception as e:
    print("❌ Failed to load Q&A data:", e)
    qa_pairs = []

# Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
question_texts = [pair["question"] for pair in qa_pairs]
question_embeddings = model.encode(question_texts, convert_to_tensor=True)

@app.post("/ask_question")
async def ask_question(req: Request):
    try:
        body = await req.json()
        question = body.get("question", "").strip()
        if not question:
            return {"answer": "Please enter a question."}

        query_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, question_embeddings, top_k=1)
        best_match_idx = hits[0][0]['corpus_id']
        best_answer = qa_pairs[best_match_idx]['answer']

        return {"answer": best_answer}
    except Exception as e:
        print("/ask_question error:", e)
        return {"answer": "Something went wrong."}

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
    return {"routine": "\n".join([l for l in lines if l])}

@app.post("/submit_feedback")
async def submit_feedback(req: Request):
    body = await req.json()
    with open("feedback_log.json", "a") as f:
        json.dump(body, f)
        f.write("\n")
    return {"status": "success"}

@app.post("/story/generate")
async def generate_story(req: Request):
    body = await req.json()
    age = body.get("age", 3)
    theme = body.get("theme", "friendship").lower()

    stories = {
        "jungle": [
            f"Once upon a time, a curious {age}-year-old monkey explored the jungle and made new friends."
        ],
        "friendship": [
            f"A {age}-year-old bear invited everyone to a party in the forest and they all had fun."
        ],
        "default": [
            f"A magical cloud danced across the sky just for a {age}-year-old child."
        ]
    }
    return {"story": choice(stories.get(theme, stories["default"]))}

@app.post("/auth/verify")
async def verify_token(req: Request):
    try:
        token = (await req.json()).get("token", "")
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        return {"status": "success", "uid": uid, "email": email}
    except Exception as e:
        print("Token verification error:", e)
        return {"status": "error", "message": str(e)}
