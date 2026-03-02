from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ load once globally (important for speed)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# JD ↔ Transcript semantic match
# --------------------------------------------------
def compute_speech_jd_match(jd_text: str, transcript: str) -> float:
    if not transcript.strip():
        return 0.0

    jd_emb = embedder.encode([jd_text])
    tr_emb = embedder.encode([transcript])
    score = cosine_similarity(jd_emb, tr_emb)[0][0]
    return float(score)


# --------------------------------------------------
# Final composite score
# --------------------------------------------------
def compute_final_score(resume_score, speech_score, communication_score):
    return (
        0.50 * resume_score
        + 0.30 * speech_score
        + 0.20 * communication_score
    )


# --------------------------------------------------
# Percentile assignment
# --------------------------------------------------
def assign_percentiles(candidates):
    total = len(candidates)
    for i, c in enumerate(candidates):
        c["percentile"] = round(100 * (total - i) / total, 2)
    return candidates


# --------------------------------------------------
# Shortlisting
# --------------------------------------------------
def shortlist_candidates(candidates, top_k=5):
    return candidates[:top_k]