from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer
import json, torch

app = FastAPI()

# ✅ Use lightweight embedding model to reduce memory
model = SentenceTransformer('paraphrase-albert-small-v2')

# ✅ Simple language translation using HuggingFace MarianMT
def translate_text(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)

    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
    translated = translation_model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ✅ Load QA Data
with open("data.json", "r") as f:
    data = json.load(f)

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

    # Translate to English if needed
    if lang != "en":
        try:
            q_translated = translate_text(q, lang, "en")
        except:
            return {"answer": "Translation error. Language not supported."}
    else:
        q_translated = q

    # Embed & find best answer
    query_embedding = model.encode(q_translated, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_match = qa_pairs[scores.argmax().item()]["answer"]

    # Translate back if needed
    if lang != "en":
        try:
            best_match = translate_text(best_match, "en", lang)
        except:
            return {"answer": "Translation error after answer lookup."}

    return {"answer": best_match}

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
