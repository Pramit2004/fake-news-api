from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect, LangDetectException
import joblib

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input schema
class InputText(BaseModel):
    inputs: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake News Detection API"}

# Predict endpoint
@app.post("/predict")
def predict(item: InputText):
    text = item.inputs.strip()

    # Language detection
    try:
        language = detect(text)
    except LangDetectException:
        raise HTTPException(
            status_code=400,
            detail="Could not detect language. Try a longer input."
        )

    if language != "en":
        return {
            "prediction": None,
            "confidence": None,
            "language": language,
            "status": "non-english",
            "message": "Model supports English language only."
        }
    
    # ✅ FIXED: Proper indentation
    if len(text.split()) < 10:
        return {
            "prediction": None,
            "confidence": None,
            "language": language,
            "status": "too_short",
            "message": "Input text too short for meaningful prediction."
        }

    # Vectorize and predict
    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    confidence_fake = float(proba[1])  # Assuming class 1 = FAKE
    label = "FAKE" if confidence_fake >= 0.5 else "REAL"

    return {
        "prediction": label,
        "confidence": round(confidence_fake, 4),
        "language": language,
        "status": "success",
        "message": "Prediction successful."
    }
