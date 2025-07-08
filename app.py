from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ✅ This line is mandatory
from pydantic import BaseModel
import joblib, re

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

# ✅ Add middleware immediately after app init
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HFStyleRequest(BaseModel):
    inputs: str

def preprocess(text):
    stop_words = set([
        'the', 'a', 'an', 'and', 'is', 'be', 'to', 'of', 'in', 'on', 'with',
        'that', 'for', 'it', 'as', 'was', 'are', 'at', 'by', 'from'
    ])
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)

@app.post("/predict")
def predict(data: HFStyleRequest):
    clean_text = preprocess(data.inputs)
    vec = vectorizer.transform([clean_text])
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec)[0][pred]
    return {
        "label": "fake" if pred == 1 else "real",
        "confidence": round(float(conf), 4)
    }

@app.get("/")
def root():
    return {
        "status": "API is live",
        "message": "POST to /predict with { inputs: 'your text' }"
    }
