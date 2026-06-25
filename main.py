import os
import glob
import re
import base64
import tempfile
from datetime import datetime

import yt_dlp
import openai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "pgadri/my-captures")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


class CaptureRequest(BaseModel):
    url: str


def download_audio(url: str, output_dir: str) -> tuple[str, str]:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Untitled")
        uploader = info.get("uploader", "Unknown")
        return title, uploader


def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text


def push_to_github(filename: str, content: str) -> str:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    encoded = base64.b64encode(content.encode()).decode()
    response = requests.put(url, headers=headers, json={
        "message": f"capture: {filename}",
        "content": encoded,
    })
    if response.status_code not in (200, 201):
        raise Exception(f"GitHub push failed: {response.text}")
    return f"https://github.com/{GITHUB_REPO}/blob/main/{filename}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/capture")
async def capture(req: CaptureRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            title, uploader = download_audio(req.url, tmpdir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

        files = glob.glob(f"{tmpdir}/*.mp3")
        if not files:
            raise HTTPException(status_code=500, detail="Audio file not found after download")

        try:
            transcript = transcribe(files[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        date = datetime.now().strftime("%Y-%m-%d")
        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-")[:60]
        filename = f"{date}-{safe_title}.md"

        note = f"""# {title}
**Source:** {req.url}
**Creator:** {uploader}
**Captured:** {datetime.now().strftime("%B %d, %Y")}

---

## Transcript

{transcript}
"""

        try:
            note_url = push_to_github(filename, note)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GitHub push failed: {str(e)}")

        return {"success": True, "title": title, "note_url": note_url}
