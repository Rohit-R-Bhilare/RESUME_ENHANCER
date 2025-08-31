from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins. Restrict later if needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class ResumeRequest(BaseModel):
    resume_text: str
    job_description: str

# Response model
class ATSResponse(BaseModel):
    score: int
    missing_keywords: List[str]
    recommendation: str

def clean_text(text: str) -> str:
    """Lowercase & remove non-alphabetic chars"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text

def extract_keywords(text: str) -> List[str]:
    """Extract keywords (nouns & proper nouns)"""
    doc = nlp(text)
    keywords = [
        token.lemma_
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and len(token) > 2
    ]
    return list(set(keywords))

@app.post("/score", response_model=ATSResponse)
def score_resume(data: ResumeRequest):
    resume_text = clean_text(data.resume_text)
    jd_text = clean_text(data.job_description)

    # Extract keywords
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    # Find missing keywords from job description
    missing = [kw for kw in jd_keywords if kw not in resume_keywords]

    # Calculate similarity score
    tfidf = TfidfVectorizer().fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    score = int(similarity * 100)

    # Recommendation
    recommendation = (
        "Consider adding these missing keywords to your resume: "
        + ", ".join(missing[:10])
        if missing
        else "Your resume matches well with the job description!"
    )

    return ATSResponse(
        score=score,
        missing_keywords=missing[:10],
        recommendation=recommendation
    )
