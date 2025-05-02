from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sentence_transformers import SentenceTransformer, util
import json
from random import choice
import torch

app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "✅ ParentWise backend is running."}

@app.head("/")
def head_root():
    return PlainTextResponse("OK")

# Load Q&A data
qa_pairs = []
try:
    with open("app/data.json", "r") as f:
        data = json.load(f)
        for topic in data.get("toddler_care", {}):
            qa_pairs.extend(data["toddler_care"][topic])
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
except Exception as e:
    print("❌ Failed to load Q&A data:", e)
    qa_pairs = []

# Lazy model load
model = None
question_embeddings = None

@app.on_event("startup")
def init_model():
    global model, question_embeddings
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        questions = [pair["question"] for pair in qa_pairs]
        question_embeddings = model.encode(questions, convert_to_tensor=True)
        print("✅ Model and embeddings ready.")
    except Exception as e:
        print("❌ Model init error:", e)

@app.post("/ask_question")
async def ask_question(req: Request):
    try:
        body = await req.json()
        q = body.get("question", "").strip()

        if not q:
            return {"answer": "Please enter a question."}
        if not model or not question_embeddings:
            return {"answer": "Model is still loading. Try again soon."}

        q_emb = model.encode(q, convert_to_tensor=True)
        result = util.semantic_search(q_emb, question_embeddings, top_k=1)
        match = result[0][0]["corpus_id"]
        return {"answer": qa_pairs[match]["answer"]}
    except Exception as e:
        print("❌ Q&A error:", e)
        return {"answer": "Something went wrong. Try again."}

@app.post("/generate_schedule")
async def generate_schedule(req: Request):
    body = await req.json()
    age = body.get("age", 2)
    wake = body.get("wake_time", "7:00 AM")
    nap = body.get("nap_pref", "1 nap")
    meals = body.get("meals", 3)

    parts = [
        f"Wake up at {wake}, then breakfast.",
        "Morning activity or learning.",
        "Mid-morning nap." if nap == "2 naps" else "",
        "Lunch and story time.",
        "Afternoon nap." if nap in ["1 nap", "2 naps"] else "",
        "Outdoor play and bonding.",
        "Dinner and quiet time.",
        "Bedtime around 8 PM."
    ]
    routine = "\n".join([line for line in parts if line])
    return {"routine": routine}

@app.post("/story/generate")
async def generate_story(req: Request):
    body = await req.json()
    age = body.get("age", 3)
    theme = body.get("theme", "friendship").lower()

    stories = {
        "jungle": [
            f"A curious {age}-year-old monkey explored the jungle and made amazing friends.",
            f"In a vibrant jungle, a baby elephant discovered a magical waterfall with her pals."
        ],
        "friendship": [
            f"A little bunny made a new friend by sharing carrots and laughter.",
            f"A {age}-year-old bear hosted a forest picnic — and made friends forever."
        ],
        "space": [
            f"A {age}-year-old astronaut launched into space and found a glittery alien buddy!",
            f"Stars twinkled as a young explorer floated by Saturn and waved to new space friends."
        ],
        "default": [
            f"A cloud danced through the sky to delight a {age}-year-old dreamer like you.",
            f"A friendly star whispered a bedtime tale to a sleepy {age}-year-old child."
        ]
    }

    return {"story": choice(stories.get(theme, stories["default"]))}

@app.post("/submit_feedback")
async def submit_feedback(req: Request):
    body = await req.json()
    try:
        with open("feedback_log.json", "a") as f:
            json.dump(body, f)
            f.write("\n")
        return {"status": "success"}
    except Exception as e:
        print("❌ Feedback error:", e)
        return {"status": "error", "message": "Failed to log feedback."}

@app.post("/auth/verify")
async def verify_token(req: Request):
    token = (await req.json()).get("token", "")
    if token == "demo-token":
        return {"status": "success", "user": "demo"}
    return {"status": "error", "message": "Invalid token"}
