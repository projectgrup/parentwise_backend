from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import json
from random import choice

app = FastAPI()

# ✅ Enable CORS for Streamlit frontend
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

# ✅ Load data.json (Q&A)
try:
    with open("data.json", "r") as f:
        data = json.load(f)

    qa_pairs = []
    for topic in data.get("toddler_care", {}):
        qa_pairs.extend(data["toddler_care"][topic])
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
except Exception as e:
    print("❌ Could not load data.json:", e)
    qa_pairs = []

# ✅ Lazy load model and embeddings
model = None
question_embeddings = None

@app.on_event("startup")
def load_model():
    global model, question_embeddings
    if qa_pairs:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        question_texts = [pair["question"] for pair in qa_pairs]
        question_embeddings = model.encode(question_texts, convert_to_tensor=True)
        print("✅ Model and embeddings initialized.")

# ✅ Semantic Q&A Endpoint
@app.post("/ask_question")
async def ask_question(req: Request):
    global model, question_embeddings
    try:
        body = await req.json()
        question = body.get("question", "").strip()

        if not question:
            return {"answer": "Please enter a question."}
        if model is None or question_embeddings is None:
            return {"answer": "Model is still loading. Please try again in a few seconds."}

        query_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, question_embeddings, top_k=1)
        best_match_idx = hits[0][0]['corpus_id']
        best_answer = qa_pairs[best_match_idx]['answer']

        return {"answer": best_answer}
    except Exception as e:
        print("❌ /ask_question error:", e)
        return {"answer": "Something went wrong. Please try again later."}

# ✅ Schedule Planner Endpoint
@app.post("/generate_schedule")
async def generate_schedule(req: Request):
    body = await req.json()
    age = body.get("age", 2)
    wake = body.get("wake_time", "7:00 AM")
    nap = body.get("nap_pref", "1 nap")
    meals = body.get("meals", 3)

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

# ✅ Story Generator Endpoint
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
        "space": [
            f"A brave {age}-year-old astronaut launched into space and discovered a friendly alien planet."
        ],
        "default": [
            f"A magical cloud danced across the sky just for a {age}-year-old child."
        ]
    }

    return {"story": choice(stories.get(theme, stories["default"]))}

# ✅ Feedback Collector Endpoint
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
        return {"status": "error", "message": "Failed to save feedback"}
