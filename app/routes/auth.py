
from fastapi import APIRouter, Form
from firebase_admin import auth

router = APIRouter()

@router.post("/verify")
def verify_token(token: str = Form(...)):
    try:
        decoded = auth.verify_id_token(token)
        return {"uid": decoded["uid"]}
    except Exception as e:
        return {"error": str(e)}
