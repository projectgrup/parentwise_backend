from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sentence_transformers import SentenceTransformer, util
import json
from random import choice

app = FastAPI()

# ✅ Enable CORS (for Streamlit frontend to work)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root health check to prevent 502
@app.get("/")
def read_root():
    return {"message": "✅ ParentWise backend is running."}

@app.head("/")
def head_root():
    return PlainTextResponse("OK")

# ✅ Load Q&A data
qa_pairs = []
try:
    with open("app/data.json", "r") as f:  # Adjust path if needed
        data = json.load(f)
        for topic in data.get("toddler_care", {}):
            qa_pairs.extend(data["toddler_care"][topic])
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
except Exception as e:
    print("❌ Could not load app/data.json:", e)
    qa_pairs = [{"question": "What is toddler care?", "answer": "Toddler care includes routines, naps, food, and love."}]

# ✅ Lazy model and embeddings
model = None
question_embeddings = None

@app.on_event("startup")
def load_model():
    global model, question_embeddings
    try:
        if qa_pairs:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            question_texts = [pair["question"] for pair in qa_pairs]
            question_embeddings = model.encode(question_texts, convert_to_tensor=True)
            print("✅ Model and embeddings loaded.")
    except Exception as e:
        print("❌ Model load failed:", e)

# ✅ Parenting Q&A route
@app.post("/ask_question")
async def ask_question(req: Request):
    global model, question_embeddings
    try:
        body = await req.json()
        question = body.get("question", "").strip()

        if not question:
            return {"answer": "Please enter a question."}
        if model is None or question_embeddings is None:
            return {"answer": "Model is still loading. Try again shortly."}

        query_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, question_embeddings, top_k=1)
        best_idx = hits[0][0]["corpus_id"]
        return {"answer": qa_pairs[best_idx]["answer"]}
    except Exception as e:
        print("❌ /ask_question error:", e)
        return {"answer": "Something went wrong. Try again later."}

# ✅ Schedule generator
@app.post("/generate_schedule")
async def generate_schedule(req: Request):
    body = await req.json()
    age = body.get("age", 2)
    wake = body.get("wake_time", "7:00 AM")
    nap = body.get("nap_pref", "1 nap")
    meals = body.get("meals", 3)

    lines = [
        f"Wake up at {wake}, then breakfast.",
        "Morning play or learning time.",
        "Mid-morning nap." if nap == "2 naps" else "",
        "Lunch and story time.",
        "Afternoon nap." if nap in ["1 nap", "2 naps"] else "",
        "Outdoor play and bonding activities.",
        "Dinner and calming routine.",
        "Bedtime around 8 PM."
    ]
    routine = "\n".join([line for line in lines if line])
    return {"routine": routine}

# ✅ Story generation
@app.post("/story/generate")
async def generate_story(req: Request):
    body = await req.json()
    age = body.get("age", 3)
    theme = body.get("theme", "friendship").lower()

    stories = {
        "jungle": [
            f"Once upon a time, a curious {age}-year-old monkey explored the jungle and made new friends.",
            f"A baby tiger and a parrot helped each other find a waterfall deep in the jungle."
        ],
        "friendship": [
            f"A {age}-year-old bear invited everyone to a forest tea party. They laughed and played together.",
            f"A shy bunny met a bird who helped it make new friends at the garden."
        ],
        "space": [
            f"A brave {age}-year-old astronaut flew to space and made friends with a glowing star.",
            f"On a trip to Mars, a {age}-year-old child discovered a musical alien who loved lullabies."
        ],
        "default": [
            f"A magical cloud danced across the sky for a {age}-year-old who loved to dream.",
            f"A glowing moon whispered gentle stories to all sleepy {age}-year-olds below."
        ]
    }

    return {"story": choice(stories.get(theme, stories["default"]))}

# ✅ Feedback logging
@app.post("/submit_feedback")
async def submit_feedback(req: Request):
    body = await req.json()
    try:
        with open("feedback_log.json", "a") as f:
            json.dump(body, f)
            f.write("\n")
        return {"status": "success"}
    except Exception as e:
        print("❌ Feedback saving failed:", e)
        return {"status": "error", "message": "Failed to save feedback."}

# ✅ Mock Firebase auth
@app.post("/auth/verify")
async def verify_token(req: Request):
    body = await req.json()
    token = body.get("token", "").strip()
    if token == "demo-token":
        return {"status": "success", "user": "demo"}
    else:
        return {"status": "error", "message": "Invalid token."}
