import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    return text.strip()


def compute_similarity(jd_text: str, resume_text: str) -> float:
    jd_emb = model.encode([jd_text])
    res_emb = model.encode([resume_text])
    score = cosine_similarity(jd_emb, res_emb)[0][0]
    return float(score)