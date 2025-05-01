from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer
import os, json

app = FastAPI()

# ✅ Use low-memory embedding model
model = SentenceTransformer("paraphrase-albert-small-v2")

# ✅ Translation with MarianMT
def translate_text(text, src_lang, tgt_lang):
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        translator_model = MarianMTModel.from_pretrained(model_name)
        tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
        translated = translator_model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception:
        return None

# ✅ Safely load data.json
try:
    file_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(file_path, "r") as f:
        data = json.load(f)
except Exception as e:
    print("❌ Failed to load data.json:", str(e))
    data = {"toddler_care": {}}

# Prepare data
qa_pairs = []
for topic in data["toddler_care"]:
    qa_pairs.extend(data["toddler_care"][topic])

questions = [item["question"] for item in qa_pairs]
question_embeddings = model.encode(questions, convert_to_tensor=True)

@app.post("/ask_question")
async def ask_question(req: Request):
    body = await req.json()
    q = body["question"]
    lang = body.get("target_language", "en")

    # Translate input to English if needed
    q_translated = translate_text(q, lang, "en") if lang != "en" else q
    if not q_translated:
        return {"answer": "⚠️ Translation to English failed. Please try another language."}

    # Retrieve best match
    query_embedding = model.encode(q_translated, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_answer = qa_pairs[scores.argmax().item()]["answer"]

    # Translate back to user language
    final_answer = translate_text(best_answer, "en", lang) if lang != "en" else best_answer
    if not final_answer:
        return {"answer": "⚠️ Translation to your language failed. Try again."}

    return {"answer": final_answer}

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
