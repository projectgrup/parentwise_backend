from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 🔐 Import Firebase initialization
from app.config.firebase_config import initialize_firebase

initialize_firebase()

# 🧠 Create the FastAPI app
app = FastAPI()

# 🌐 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

