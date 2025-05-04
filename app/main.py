from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from random import choice
import json
from app.core import rag  # ✅ Import custom RAG module

app = FastAPI()

# ✅ Enable CORS for frontend communication
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

# ✅ Load model + data on server startup
@app.on_event("startup")
def on_startup():
    rag.load_qa_data()
    rag.load_model_and_index()

# ✅ Parent Q&A endpoint (used internally)
@app.post("/ask_question")
async def ask_question(req: Request):
    try:
        body = await req.json()
        q = body.get("question", "").strip()

        if not q:
            return {"answer": "Please enter a question."}

        answer = rag.search_answer(q)
        return {"answer": answer}
    except Exception as e:
        print("❌ Q&A error:", e)
        return {"answer": "Something went wrong. Try again."}

# ✅ FIX for frontend — route alias for Streamlit `/ask`
@app.post("/ask")
async def ask_alias(req: Request):
    return await ask_question(req)

# ✅ Toddler Schedule Generator
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

# ✅ Story Generator
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

# ✅ Feedback Logger
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

# ✅ Firebase Token Verification (Demo only)
@app.post("/auth/verify")
async def verify_token(req: Request):
    token = (await req.json()).get("token", "")
    if token == "demo-token":
        return {"status": "success", "user": "demo"}
    return {"status": "error", "message": "Invalid token"}
