from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os
import shutil
from app.resume_ranker import extract_text_from_pdf, compute_similarity
from app.utils import generate_reasoning
from app.video_analyzer import analyze_video_speech
from fastapi import Request
import aiofiles
from app.scoring_engine import (
    compute_speech_jd_match,
    compute_final_score,
    assign_percentiles,
    shortlist_candidates,
)
from typing import List

app = FastAPI(title="ZyraHire AI Service")

UPLOAD_DIR = "uploads/resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


VIDEO_DIR = "uploads/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


@app.get("/")
def health():
    return {"status": "AI service running"}


@app.post("/rank-resumes", summary="Upload multiple resume PDFs")
async def rank_resumes(
    jd_text: str = Form(...),
    files: List[UploadFile] = File(..., media_type="application/pdf")
):
    results = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        resume_text = extract_text_from_pdf(file_path)
        score = compute_similarity(jd_text, resume_text)

        # 🔥 safe reasoning (won’t crash if key missing)
        try:
            reason = generate_reasoning(jd_text, resume_text)
        except Exception:
            reason = "AI reasoning unavailable"

        results.append({
            "resume": file.filename,
            "score": round(score, 4),
            "reasoning": reason
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {"ranked_candidates": results}
    

@app.post("/analyze-video")
async def analyze_video(
    jd_text: str = Form(...),
    file: UploadFile = File(...)
):
    video_path = f"uploads/videos/{file.filename}"
    os.makedirs("uploads/videos", exist_ok=True)

    # ✅ SAFE async write
    async with aiofiles.open(video_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    result = analyze_video_speech(video_path, jd_text)

    return result
@app.post("/evaluate-candidates")
async def evaluate_candidates(
    jd_text: str = Form(...),
    files: List[UploadFile] = File(...)
):
    candidates = []
    os.makedirs("uploads/videos", exist_ok=True)

    for file in files:
        # -----------------------------
        # 1. Save video
        # -----------------------------
        video_path = f"uploads/videos/{file.filename}"

        async with aiofiles.open(video_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # -----------------------------
        # 2. Run video analysis
        # -----------------------------
        video_result = analyze_video_speech(video_path, jd_text)

        transcript = video_result["transcript"]
        communication_score = video_result["communication_score"]

        # ✅ USE PRECOMPUTED relevance
        speech_score = video_result["speech_relevance"]

        # -----------------------------
        # 3. TEMP resume score
        # -----------------------------
        resume_score = 0.5  # placeholder

        # -----------------------------
        # 4. Final score
        # -----------------------------
        final_score = compute_final_score(
            resume_score,
            speech_score,
            communication_score,
        )

        candidates.append({
            "candidate": file.filename,
            "resume_score": round(resume_score, 4),
            "speech_jd_score": round(speech_score, 4),
            "communication_score": round(communication_score, 4),
            "final_score": round(final_score, 4),
        })

    # -----------------------------
    # 5. Rank candidates
    # -----------------------------
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # -----------------------------
    # 6. Assign percentiles
    # -----------------------------
    candidates = assign_percentiles(candidates)

    # -----------------------------
    # 7. Shortlist
    # -----------------------------
    shortlisted = shortlist_candidates(candidates, top_k=5)

    return {
        "total_candidates": len(candidates),
        "shortlisted_count": len(shortlisted),
        "shortlisted_candidates": shortlisted,
        "full_ranking": candidates,
    }