from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def read_root():
    return {"message": "Welcome to ParentWise AI - FastAPI is running ✅"}

@app.post("/qa/ask")
def ask_question(question: str = Form(...), lang: str = Form("en")):
    return {
        "response": f"👶 Hello! Thanks for your question in {lang}. For toddlers, soft fruits, cereals, and veggies are great!"
    }

