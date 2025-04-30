from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials
from app.routes import qa, feedback, auth

cred = credentials.Certificate("firebase_key.json")  # Upload your JSON here
firebase_admin.initialize_app(cred)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

app.include_router(qa.router, prefix="/qa")
app.include_router(feedback.router, prefix="/feedback")
app.include_router(auth.router, prefix="/auth")

@app.get("/")
def home():
    return {"message": "Welcome to ParentWise API"}


