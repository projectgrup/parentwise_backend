from fastapi import APIRouter, Form
from firebase_admin import firestore

router = APIRouter()
db = firestore.client()

@router.post("/submit")
def submit_feedback(question: str = Form(...), answer: str = Form(...), rating: int = Form(...)):
    db.collection("feedback").add({
        "question": question,
        "answer": answer,
        "rating": rating
    })
    return {"status": "saved"}

