from fastapi import APIRouter, Form
from transformers import pipeline

router = APIRouter()

@router.post("/generate")
def generate_story(age: str = Form(...), theme: str = Form(...)):
    # ✅ Load the model only when this route is called
    story_generator = pipeline('text-generation', model='distilgpt2')
    prompt = f"Tell a bedtime story for a {age}-year-old about {theme}."
    story = story_generator(prompt, max_length=250, do_sample=True, temperature=0.7)
    return {"story": story[0]['generated_text']}

