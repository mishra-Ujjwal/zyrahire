import os
from groq import Groq

def generate_reasoning(jd_text: str, resume_text: str) -> str:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""
You are an expert technical recruiter.

Job Description:
{jd_text}

Candidate Resume:
{resume_text[:4000]}

Give a short professional evaluation covering:
- match level
- key strengths
- missing skills

Keep it under 80 words.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ GROQ ERROR:", str(e))  # 👈 IMPORTANT
        raise