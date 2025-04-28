
from fastapi import APIRouter, Form

router = APIRouter()

@router.post("/generate")
def generate_schedule(age: str = Form(...), wake_time: str = Form(...), sleep_time: str = Form(...)):
    routine = {
        "Morning": f"{wake_time} - Wake up, breakfast, hygiene",
        "Midday": "Playtime, creative activity",
        "Afternoon": "Lunch, nap, quiet time",
        "Evening": f"Dinner, storytelling, bedtime by {sleep_time}"
    }
    return {"age": age, "routine": routine}
