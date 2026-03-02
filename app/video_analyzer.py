import os
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# lightweight whisper model (fast CPU friendly)
whisper_model = WhisperModel("base", compute_type="int8")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_audio(video_path: str, audio_path: str):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def transcribe_audio(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path)
    text = " ".join([seg.text for seg in segments])
    return text.strip()

# -----------------------------
# communication intelligence
# -----------------------------

FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically",
    "actually", "literally", "kind of", "sort of"
}

def compute_fluency_score(transcript: str, duration: float):
    words = transcript.lower().split()
    word_count = max(1, len(words))

    wps = word_count / max(duration, 1)

    if 1.5 <= wps <= 3.5:
        pace_score = 1.0
    elif 1.0 <= wps < 1.5 or 3.5 < wps <= 4.5:
        pace_score = 0.7
    else:
        pace_score = 0.4

    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_ratio = filler_count / word_count

    if filler_ratio < 0.02:
        filler_score = 1.0
    elif filler_ratio < 0.05:
        filler_score = 0.7
    else:
        filler_score = 0.4

    return 0.6 * pace_score + 0.4 * filler_score


def compute_vocab_score(transcript: str):
    words = transcript.lower().split()
    if not words:
        return 0.0

    unique_ratio = len(set(words)) / len(words)

    if unique_ratio > 0.6:
        return 1.0
    elif unique_ratio > 0.45:
        return 0.7
    else:
        return 0.4


def compute_speech_jd_match(jd_text: str, transcript: str):
    jd_emb = embedder.encode([jd_text])
    tr_emb = embedder.encode([transcript])
    score = cosine_similarity(jd_emb, tr_emb)[0][0]
    return float(score)

    def analyze_video_speech(video_path: str, jd_text: str):
    audio_path = video_path.replace(".mp4", ".wav")

    video = VideoFileClip(video_path)
    duration_seconds = max(video.duration, 1)

    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)

    word_count = len(transcript.split())

    fluency_score = compute_fluency_score(transcript, duration_seconds)
    vocab_score = compute_vocab_score(transcript)
    relevance_score = compute_speech_jd_match(jd_text, transcript)

    communication_score = round(
        0.4 * fluency_score +
        0.3 * vocab_score +
        0.3 * relevance_score,
        4
    )

    return {
        "transcript": transcript,
        "communication_score": communication_score,
        "speech_relevance": round(relevance_score, 4),
        "word_count": word_count,
    }