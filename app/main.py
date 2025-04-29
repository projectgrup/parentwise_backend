from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ Firebase initialization
import firebase_admin
from firebase_admin import credentials

# Load Firebase admin SDK key
cred = credentials.Certificate("app/firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

# 🧠 Create the FastAPI app
app = FastAPI()

# 🌐 Enable CORS (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with your Streamlit domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🚏 Register all routes
from app.routes import qa, schedule, feedback, auth, story

app.include_router(qa.router, prefix="/qa")
app.include_router(schedule.router, prefix="/schedule")
app.include_router(feedback.router, prefix="/feedback")
app.include_router(auth.router, prefix="/auth")
app.include_router(story.router, prefix="/story")

# 🌱 Health check route
@app.get("/")
def root():
    return {"message": "Welcome to ParentWise AI!"}
