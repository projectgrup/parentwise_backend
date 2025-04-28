
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import qa, schedule, feedback, auth, story

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(qa.router, prefix="/qa")
app.include_router(schedule.router, prefix="/schedule")
app.include_router(feedback.router, prefix="/feedback")
app.include_router(auth.router, prefix="/auth")
app.include_router(story.router, prefix="/story")

@app.get("/")
def root():
    return {"message": "Welcome to ParentWise AI!"}
