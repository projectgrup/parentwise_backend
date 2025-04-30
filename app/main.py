from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.firebase_config import initialize_firebase
from app.routes import qa, feedback, auth

# ✅ Securely initialize Firebase
initialize_firebase()

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



