from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect, LangDetectException
import joblib

# Create FastAPI app
app = FastAPI()

# Enable CORS (you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model = joblib.load("model/logistic_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

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

    # Detect language
    try:
        language = detect(text)
    except LangDetectException:
        raise HTTPException(status_code=400, detail="Could not detect language. Try a longer input.")

    if language != "en":
        return {
            "prediction": None,
            "language": language,
            "status": "non-english",
            "message": "Model supports English language only."
        }

    # Perform prediction
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return {
        "prediction": bool(pred),
        "language": language,
        "status": "success",
        "message": "Prediction successful."
    }
