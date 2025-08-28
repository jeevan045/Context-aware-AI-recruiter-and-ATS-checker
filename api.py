# api.py
# ------------------------------------------------------------
# Context-Aware AI Recruiter: API Backend
# ------------------------------------------------------------
import io
import re
import warnings
import os # <--- THIS LINE IS THE FIX
from typing import List, Dict, Tuple

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# PDF parsing
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
from PyPDF2 import PdfReader

# NLP + Embeddings
import nltk
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

# One-time downloads
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# Configuration
SKILL_KB = sorted(set([
    "python", "java", "c++", "javascript", "typescript", "sql", "scala", "go", "r",
    "machine learning", "deep learning", "data analysis", "data visualization",
    "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "pytorch", "nlp", "aws"
]))

# --- Model loads ---
def load_models():
    """Loads and returns the NLP models."""
    print("Loading spaCy model...")
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
        exit()
    print("Loading SentenceTransformer model...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Models loaded successfully.")
    return nlp_model, sbert_model

nlp, sbert = load_models()
STOPWORDS = set(stopwords.words("english"))

# --- Utility Functions ---

def read_pdf(file: bytes) -> str:
    text = ""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(file)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
        except Exception:
            text = ""
    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(file))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception:
            return ""
    return text

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def preprocess_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\.\+\#\s\-]", " ", s)
    tokens = [t for t in s.split() if t not in STOPWORDS]
    return " ".join(tokens)

def extract_skills(text: str) -> List[str]:
    text_l = text.lower()
    found = set()
    for sk in SKILL_KB:
        if re.search(rf"(?<![a-z0-9]){re.escape(sk)}(?![a-z0-9])", text_l):
            found.add(sk)
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        c = chunk.text.lower().strip()
        if c in SKILL_KB:
            found.add(c)
    return sorted(found)

def extract_years_experience(text: str) -> float:
    yrs = []
    for m in re.finditer(r"(\d+(\.\d+)?)\s+\+?\s*(years|yrs)\s+(of\s+)?(experience|exp)", text, re.I):
        try:
            yrs.append(float(m.group(1)))
        except Exception:
            pass
    if yrs:
        return float(np.clip(np.median(yrs), 0, 15))
    return 0.0

def jd_required_skills(jd_text: str, top_n: int = 15) -> List[str]:
    jd_p = preprocess_text(jd_text)
    counts = {}
    for sk in SKILL_KB:
        hits = len(re.findall(rf"(?<![a-z0-9]){re.escape(sk)}(?![a-z0-9])", jd_p))
        if hits > 0:
            counts[sk] = hits
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]

def embedding_similarity(a: str, b: str) -> float:
    ea = sbert.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    eb = sbert.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(ea, eb).cpu().numpy()[0][0])

def composite_score(semantic: float, resume_skills: List[str], req_skills: List[str], years_exp: float) -> Tuple[float, Dict[str, float]]:
    w_semantic, w_skills, w_exp = 0.6, 0.3, 0.1
    
    skill_set = set(resume_skills)
    skill_matches = sum(1 for sk in req_skills if sk in skill_set)
    skill_cov = skill_matches / len(req_skills) if req_skills else 0.0
    
    exp_norm = min(years_exp / 5.0, 1.0)

    total = (w_semantic * semantic) + (w_skills * skill_cov) + (w_exp * exp_norm)
    details = {"semantic": semantic, "skills_coverage": skill_cov, "exp_norm": exp_norm}
    return float(total), details


# --- Flask App ---
app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'resumeFile' not in request.files:
        return jsonify({'error': 'No resume file provided'}), 400
    
    job_description = request.form.get('jobDescription', '')
    resume_file = request.files['resumeFile']

    if not job_description.strip():
        return jsonify({'error': 'Job description is empty'}), 400

    try:
        resume_bytes = resume_file.read()
        resume_text = read_pdf(resume_bytes)
        
        jd_clean = preprocess_text(job_description)
        resume_clean = preprocess_text(resume_text)

        req_skills = jd_required_skills(job_description)
        resume_skills = extract_skills(resume_text)
        years_exp = extract_years_experience(resume_text)

        semantic_sim = embedding_similarity(resume_clean, jd_clean)
        score, parts = composite_score(semantic_sim, resume_skills, req_skills, years_exp)

        missing_skills = [s for s in req_skills if s not in set(resume_skills)]
        
        response_data = {
            "matchScore": round(score * 100, 1),
            "summary": f"The candidate has {years_exp} years of experience and matches {len(resume_skills)} skills. The semantic similarity with the JD is {semantic_sim:.2f}.",
            "pros": resume_skills,
            "cons": missing_skills,
            "semantic_score": parts["semantic"],
            "skills_score": parts["skills_coverage"],
            "experience_score": parts["exp_norm"]
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Failed to process the resume.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
