import os
import glob
import re
import json
import base64
import uuid as _uuid
import tempfile
import secrets
import random
from datetime import datetime, timedelta, timezone

import yt_dlp
import openai
import requests
import asyncpg
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "pgadri/my-captures")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
FROM_EMAIL = "hello@vibecoded.tech"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30
OTP_TTL = 600       # 10 minutes
OTP_MAX_ATTEMPTS = 3

# ─── DB pool ─────────────────────────────────────────────────────────────────

_pool: asyncpg.Pool | None = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            ssl="require",
            min_size=1,
            max_size=5,
        )
    return _pool

# ─── JWT helpers ─────────────────────────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)

def create_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS)
    return jwt.encode({"sub": user_id, "exp": exp}, SESSION_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, SESSION_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None

async def current_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    pool: asyncpg.Pool = Depends(get_pool),
) -> str:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    uid = decode_token(creds.credentials)
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return uid

# ─── Auth models ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class SigninRequest(BaseModel):
    email: str
    password: str

class OTPVerifyRequest(BaseModel):
    email: str
    code: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

# ─── Redis helpers (Upstash REST) ─────────────────────────────────────────────

def _redis_headers():
    return {"Authorization": f"Bearer {REDIS_TOKEN}"}

def redis_set(key: str, value: str, ex: int = OTP_TTL):
    requests.post(f"{REDIS_URL}/set/{key}/{value}/ex/{ex}", headers=_redis_headers(), timeout=5)

def redis_get(key: str) -> str | None:
    r = requests.get(f"{REDIS_URL}/get/{key}", headers=_redis_headers(), timeout=5)
    return r.json().get("result")

def redis_del(key: str):
    requests.post(f"{REDIS_URL}/del/{key}", headers=_redis_headers(), timeout=5)

def redis_incr(key: str) -> int:
    r = requests.post(f"{REDIS_URL}/incr/{key}", headers=_redis_headers(), timeout=5)
    return r.json().get("result", 0)

# ─── OTP helpers ─────────────────────────────────────────────────────────────

def generate_otp() -> str:
    return f"{random.randint(0, 999999):06d}"

def otp_key(email: str, kind: str) -> str:
    return f"otp:{kind}:{email.lower()}"

def attempts_key(email: str, kind: str) -> str:
    return f"otp_attempts:{kind}:{email.lower()}"

def send_otp_email(to_email: str, name: str, code: str, kind: str = "verify"):
    if kind == "verify":
        subject = "Your Vibecoded verification code"
        body = f"""
        <div style="font-family:sans-serif;max-width:480px;margin:auto">
          <h2 style="color:#2A1B5E">Verify your email</h2>
          <p>Hi {name}, welcome to Vibecoded.</p>
          <p>Your verification code is:</p>
          <div style="font-size:36px;font-weight:800;letter-spacing:8px;color:#2A1B5E;padding:20px 0">{code}</div>
          <p style="color:#888">Valid for 10 minutes. Don't share this code with anyone.</p>
        </div>"""
    else:
        subject = "Reset your Vibecoded password"
        body = f"""
        <div style="font-family:sans-serif;max-width:480px;margin:auto">
          <h2 style="color:#2A1B5E">Reset your password</h2>
          <p>Hi {name}, we received a request to reset your password.</p>
          <p>Your reset code is:</p>
          <div style="font-size:36px;font-weight:800;letter-spacing:8px;color:#2A1B5E;padding:20px 0">{code}</div>
          <p style="color:#888">Valid for 10 minutes. If you didn't request this, ignore this email.</p>
        </div>"""

    requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
        json={"from": f"Vibecoded <{FROM_EMAIL}>", "to": [to_email], "subject": subject, "html": body},
        timeout=10,
    )


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


class CaptureRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    question: str
    captures: list[dict]


class MapPushRequest(BaseModel):
    title: str
    description: str
    author: str
    forked_from: str | None = None
    capture_count: int = 0


class ScanRepoRequest(BaseModel):
    repo_url: str


class AnalyzeImageRequest(BaseModel):
    image_base64: str


class PasteTranscriptRequest(BaseModel):
    text: str
    title: str = "Untitled"
    source_url: str | None = None
    platform: str | None = None


class ReputationSyncRequest(BaseModel):
    points: int


class CreatorApplicationRequest(BaseModel):
    motivation: str
    sample_content: str


class CreatePacketRequest(BaseModel):
    title: str
    description: str = ''
    category: str = 'founder'
    cover_emoji: str = '📦'


class UpdatePacketRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None
    cover_emoji: str | None = None


class CreateChapterRequest(BaseModel):
    title: str
    content: str
    chapter_order: int = 0
    is_preview: bool = False


class UpdateChapterRequest(BaseModel):
    title: str | None = None
    content: str | None = None
    chapter_order: int | None = None
    is_preview: bool | None = None


class AdminReviewRequest(BaseModel):
    action: str  # "approve" or "reject"
    reason: str | None = None


ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")


def require_admin(request: Request):
    secret = request.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Admin access required")


def extract_youtube_id(url: str) -> str | None:
    m = re.search(r'(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})', url)
    return m.group(1) if m else None


def get_youtube_transcript(url: str) -> tuple[str, str, str] | None:
    """Try to get YouTube transcript without downloading audio. Returns (title, uploader, transcript) or None."""
    vid = extract_youtube_id(url)
    if not vid:
        return None
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
        try:
            segments = YouTubeTranscriptApi.get_transcript(vid)
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        transcript = ' '.join(s['text'] for s in segments)
        # Fetch metadata without downloading
        info_opts = {'quiet': True, 'skip_download': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info.get('title', 'Untitled'), info.get('uploader', 'Unknown'), transcript
    except Exception:
        return None


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
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Untitled")
            uploader = info.get("uploader", "Unknown")
            return title, uploader
    except yt_dlp.utils.ExtractorError as e:
        msg = str(e).lower()
        if "login" in msg or "private" in msg or "age" in msg:
            raise Exception("This video is private or requires login. Try a public video.")
        if "copyright" in msg or "removed" in msg or "unavailable" in msg:
            raise Exception("This video is unavailable or has been removed.")
        raise Exception(f"Could not download video. The platform may have blocked this request.")
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        if "instagram" in msg or "tiktok" in msg:
            raise Exception("Instagram and TikTok sometimes block downloads. Try again in a moment, or try a different video.")
        raise Exception("Download failed. Check the URL is correct and the video is public.")


def transcribe(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            result = get_groq_client().audio.transcriptions.create(model="whisper-large-v3-turbo", file=f)
        return result.text
    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg or "rate_limit" in msg.lower():
            raise Exception("AI service is temporarily over capacity. Please try again in a minute.")
        raise


def summarize(transcript: str, raw_title: str, full_transcript: bool = False) -> dict:
    """Returns { title, concepts, actions, quotes, bullets, preview }."""
    response = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge extraction assistant for app builders and vibe coders. "
                    "Given a video transcript, extract the full value of the content. "
                    "Respond with a JSON object with exactly four keys:\n"
                    "- 'title': a specific, concrete title under 12 words — captures the real topic, not generic\n"
                    "- 'category': one word from this list that best describes the content: technical, marketing, launch, pricing, founder, product\n"
                    "- 'concepts': array of 4-6 key insights or facts from the content — real specifics, not summaries, complete sentences up to 200 chars\n"
                    "- 'actions': array of 3-5 concrete actions. For TECHNICAL content: specific commands to run, exact packages to install, or precise code changes to make (e.g. 'Run npm audit and fix all high-severity CVEs', 'Add helmet.js to set security headers', 'Validate all user inputs with zod before processing'). For non-technical: concrete next steps starting with a verb.\n"
                    "- 'quotes': array of 1-3 memorable direct quotes or sharp paraphrases from the speaker — the lines worth remembering\n"
                    "No bullet symbols. Extract genuine substance — the specific numbers, decisions, and lessons, not vague advice.\n"
                    "Use 'technical' only if the content is primarily about writing code, building systems, or engineering implementation. "
                    "Business, growth, strategy, and founder topics are NOT technical."
                ),
            },
            {
                "role": "user",
                "content": f"Raw title from video: {raw_title}\n\nTranscript:\n{transcript[:32000] if full_transcript else transcript[:8000]}",
            },
        ],
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(response.choices[0].message.content)
        concepts = [str(b) for b in data.get("concepts", [])][:6]
        actions = [str(b) for b in data.get("actions", [])][:5]
        quotes = [str(b) for b in data.get("quotes", [])][:3]
        valid_categories = {"technical", "marketing", "launch", "pricing", "founder", "product"}
        category = data.get("category", "founder").lower()
        if category not in valid_categories:
            category = "founder"
        preview = "\n".join(f"• {b}" for b in concepts)
        return {
            "title": data.get("title") or raw_title,
            "category": category,
            "concepts": concepts,
            "actions": actions,
            "quotes": quotes,
            "bullets": concepts,  # backward compat
            "preview": preview,
        }
    except Exception:
        return {"title": raw_title, "category": "founder", "concepts": [], "actions": [], "quotes": [], "bullets": [], "preview": ""}


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
@limiter.limit("10/minute")
async def capture(request: Request, req: CaptureRequest):
    title = "Untitled"
    uploader = "Unknown"
    transcript = ""
    used_transcript_api = False

    # Fast path: YouTube Transcript API (free, instant, full text)
    yt_result = get_youtube_transcript(req.url)
    if yt_result:
        title, uploader, transcript = yt_result
        used_transcript_api = True

    if not used_transcript_api:
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

    summary = summarize(transcript, title, full_transcript=used_transcript_api)
    clean_title = summary["title"]
    bullets = summary["bullets"]
    preview = "\n".join(f"• {b}" for b in bullets)

    date = datetime.now().strftime("%Y-%m-%d")
    safe_title = re.sub(r"[^\w\s-]", "", clean_title).strip().replace(" ", "-")[:60]
    filename = f"{date}-{safe_title}.md"

    note = f"""# {clean_title}
**Source:** {req.url}
**Creator:** {uploader}
**Captured:** {datetime.now().strftime("%B %d, %Y")}

---

## Key Insights

{preview}

---

## Full Transcript

{transcript}
"""

    note_url = None
    try:
        note_url = push_to_github(filename, note)
    except Exception:
        pass

    return {
        "success": True,
        "title": clean_title,
        "note_url": note_url or "",
        "preview": preview,
        "bullets": summary["bullets"],
        "category": summary["category"],
        "concepts": summary["concepts"],
        "actions": summary["actions"],
        "quotes": summary["quotes"],
        "transcript": transcript,
        "creator": uploader,
    }


@app.post("/capture/paste")
@limiter.limit("20/minute")
async def capture_paste(request: Request, req: PasteTranscriptRequest):
    """Capture from pasted text — blog posts, scripts, newsletters, notes."""
    if not req.text or len(req.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Please paste at least a few sentences of content.")

    summary = summarize(req.text, req.title, full_transcript=True)
    clean_title = summary["title"]
    bullets = summary["bullets"]
    preview = "\n".join(f"• {b}" for b in bullets)

    date = datetime.now().strftime("%Y-%m-%d")
    safe_title = re.sub(r"[^\w\s-]", "", clean_title).strip().replace(" ", "-")[:60]
    filename = f"{date}-{safe_title}.md"

    note = f"""# {clean_title}
**Source:** {req.source_url or req.platform or "Pasted text"}
**Captured:** {datetime.now().strftime("%B %d, %Y")}

---

## Key Insights

{preview}

---

## Full Content

{req.text}
"""

    note_url = None
    try:
        note_url = push_to_github(filename, note)
    except Exception:
        pass

    return {
        "success": True,
        "title": clean_title,
        "note_url": note_url or "",
        "preview": preview,
        "bullets": summary["bullets"],
        "category": summary["category"],
        "concepts": summary["concepts"],
        "actions": summary["actions"],
        "quotes": summary["quotes"],
        "transcript": req.text,
        "creator": req.platform or "Creator",
    }


@app.post("/push-map")
async def push_map(req: MapPushRequest):
    date = datetime.now().strftime("%Y-%m-%d")
    safe_title = re.sub(r"[^\w\s-]", "", req.title).strip().replace(" ", "-")[:60].lower()
    filename = f"maps/{safe_title}/README.md"

    fork_line = f"\n*Forked from {req.forked_from}*" if req.forked_from else ""

    content = f"""# {req.title}
*A Grimoire knowledge map by {req.author}*{fork_line}
*Published: {date} · {req.capture_count} captures*

## About

{req.description or "A curated collection of knowledge captures."}

## Contents

This map contains {req.capture_count} knowledge captures with expert insights, action items, and AI prompts ready for Cursor or any AI coding tool.

---
*Generated by [Grimoire](https://grimoire.app) — the knowledge platform for vibe coders*
"""

    try:
        note_url = push_to_github(filename, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub push failed: {str(e)}")

    return {"success": True, "url": note_url}


@app.post("/analyze-image")
@limiter.limit("10/minute")
async def analyze_image(request: Request, req: AnalyzeImageRequest):
    response = get_groq_client().chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a knowledge extraction assistant for app builders and vibe coders. "
                        "Extract from this screenshot: (1) a specific title under 12 words describing what this is about, "
                        "(2) 3-5 actionable bullet insights — concrete and specific, not vague summaries. "
                        "Respond with JSON only, no markdown: {\"title\": \"...\", \"bullets\": [\"...\", ...]}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"},
                },
            ],
        }],
        max_tokens=400,
    )
    try:
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        data = json.loads(raw)
        bullets = [str(b) for b in data.get("bullets", [])][:5]
        preview = "\n".join(f"• {b}" for b in bullets)
        return {
            "title": data.get("title", "Screenshot capture"),
            "preview": preview,
            "bullets": bullets,
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to analyze image")


def _parse_owner_repo(repo_url: str) -> tuple[str, str]:
    s = repo_url.strip().rstrip("/")
    for prefix in ["https://", "http://"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.startswith("github.com/"):
        s = s[len("github.com/"):]
    parts = s.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Cannot parse repo URL: {repo_url!r}. Expected format: owner/repo")
    return parts[0], parts[1]


def _gh(path: str, token: str | None = None) -> dict | list | None:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(f"https://api.github.com{path}", headers=headers, timeout=10)
    if r.status_code == 404:
        return None
    if r.status_code == 403:
        raise HTTPException(status_code=429, detail="GitHub API rate limit hit. Try again in ~1 hour.")
    r.raise_for_status()
    return r.json()


@app.post("/scan-repo")
async def scan_repo(req: ScanRepoRequest):
    try:
        owner, repo = _parse_owner_repo(req.repo_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    gh_token = os.environ.get("GITHUB_TOKEN")

    root = _gh(f"/repos/{owner}/{repo}/contents/", gh_token)
    if root is None:
        raise HTTPException(status_code=404, detail="Repo not found or is private. Only public repos are supported without OAuth.")

    root_files: dict[str, dict] = {}
    if isinstance(root, list):
        root_files = {item["name"].lower(): item for item in root}

    # Fetch package.json if present
    all_deps: dict[str, str] = {}
    pkg_data: dict = {}
    if "package.json" in root_files:
        raw = _gh(f"/repos/{owner}/{repo}/contents/package.json", gh_token)
        if raw and isinstance(raw, dict) and raw.get("content"):
            try:
                pkg_data = json.loads(base64.b64decode(raw["content"]).decode())
                all_deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}
            except Exception:
                pass

    # Detect stack
    detected_stack: list[str] = []
    if all_deps:
        if "expo" in all_deps:
            detected_stack.append("expo")
        if "react-native" in all_deps:
            detected_stack.append("react-native")
        if "next" in all_deps:
            detected_stack.append("nextjs")
        if "stripe" in all_deps or "@stripe/stripe-js" in all_deps:
            detected_stack.append("stripe")
        if "@supabase/supabase-js" in all_deps:
            detected_stack.append("supabase")
        if "firebase" in all_deps or "@firebase/app" in all_deps:
            detected_stack.append("firebase")
        if "openai" in all_deps or "@anthropic-ai/sdk" in all_deps:
            detected_stack.append("openai")
    if "requirements.txt" in root_files or "pyproject.toml" in root_files:
        detected_stack.append("fastapi")

    findings: list[dict] = []

    # CRITICAL: exposed .env files
    for env_name in [".env", ".env.local", ".env.production", ".env.development"]:
        if env_name in root_files:
            findings.append({
                "id": f"exposed-env-{env_name.replace('.', '')}",
                "title": f"{env_name} committed to repo",
                "severity": "critical",
                "category": "Security",
                "description": f"{env_name} is committed to the repo. Any secrets inside are publicly visible to anyone who finds this repository.",
                "aiPrompt": f"My {env_name} file is committed to a public repo. Help me: (1) immediately rotate every exposed credential, (2) remove the file from git history using git filter-branch or BFG Repo Cleaner, (3) add {env_name} to .gitignore, (4) create a {env_name}.example with placeholder values only.",
            })

    # HIGH: no .gitignore
    if ".gitignore" not in root_files:
        findings.append({
            "id": "no-gitignore",
            "title": "No .gitignore file",
            "severity": "high",
            "category": "Security",
            "description": "Without .gitignore, build artifacts, node_modules, and secret .env files can be accidentally committed.",
            "aiPrompt": "Create a comprehensive .gitignore for my project. Include: node_modules, all .env variants, build directories (dist, .next, .expo), OS files (.DS_Store), IDE files (.vscode, .idea), and any other generated files that should not be in version control.",
        })

    # MEDIUM: no .env.example
    has_env_example = ".env.example" in root_files or ".env.sample" in root_files
    if not has_env_example:
        findings.append({
            "id": "no-env-example",
            "title": "No .env.example file",
            "severity": "medium",
            "category": "Infrastructure",
            "description": "No .env.example found. Contributors cannot set up the project without knowing what environment variables are required.",
            "aiPrompt": "Scan my codebase for every environment variable I access (process.env.*, os.environ.get, etc). Create a .env.example listing all of them as placeholder values with a comment for each explaining what it is and how to get it.",
        })

    # HIGH: no Terms of Service
    terms_names = {"terms.md", "terms_of_service.md", "terms-of-service.md", "tos.md", "terms.txt", "terms_of_service.txt"}
    if not (terms_names & set(root_files.keys())):
        findings.append({
            "id": "no-terms",
            "title": "No Terms of Service found",
            "severity": "high",
            "category": "Legal",
            "description": "No Terms of Service file found. Required for apps that collect user data or handle payments.",
            "aiPrompt": "Generate a Terms of Service for my app. Include: acceptable use, user accounts, intellectual property, payment terms (if applicable), termination, disclaimers, and limitation of liability. Tell me where to host it and how to link it from my signup screen.",
        })

    # HIGH: no Privacy Policy
    privacy_names = {"privacy.md", "privacy_policy.md", "privacy-policy.md", "privacy.txt", "privacy_policy.txt"}
    if not (privacy_names & set(root_files.keys())):
        findings.append({
            "id": "no-privacy",
            "title": "No Privacy Policy found",
            "severity": "high",
            "category": "Legal",
            "description": "No Privacy Policy found. GDPR, CCPA, and App Store guidelines all require one if you collect user data.",
            "aiPrompt": "Generate a Privacy Policy for my app. Before writing it, ask me: what personal data I collect, what third-party services I use (analytics, payments, crash reporting), and where my users are located. Produce a GDPR/CCPA-compliant policy in markdown.",
        })

    # HIGH: no error monitoring
    if all_deps:
        sentry_pkgs = {"@sentry/react-native", "sentry-expo", "@sentry/nextjs", "@sentry/node"}
        if not (sentry_pkgs & set(all_deps.keys())):
            findings.append({
                "id": "no-error-monitoring",
                "title": "No error monitoring detected",
                "severity": "high",
                "category": "Infrastructure",
                "description": "No Sentry or error monitoring package found in package.json. Production crashes are invisible without monitoring.",
                "aiPrompt": "Add Sentry error monitoring to my project. Install the correct Sentry SDK for my stack, initialize with my DSN, wrap my root component (React Native/Expo) or configure middleware (Next.js/Node). Capture unhandled exceptions and set environment to 'production' on deploy. Show me the complete setup.",
            })

    # MEDIUM: LICENSE missing
    if "license" not in root_files and "license.md" not in root_files and "license.txt" not in root_files:
        findings.append({
            "id": "no-license",
            "title": "No LICENSE file",
            "severity": "medium",
            "category": "Legal",
            "description": "No LICENSE file found. Without a license, the default copyright applies — no one can legally use, copy, or contribute to your code.",
            "aiPrompt": "What open-source license should I use for my project? Tell me the differences between MIT, Apache 2.0, and GPL, and generate the correct LICENSE file for my situation. If this is a commercial product I want to keep private, explain the alternative.",
        })

    # Calculate score
    penalty_map = {"critical": 25, "high": 15, "medium": 8}
    score = 100
    for f in findings:
        score -= penalty_map.get(f["severity"], 5)
    score = max(0, score)

    return {
        "owner": owner,
        "repo": repo,
        "scannedAt": datetime.now().isoformat(),
        "score": score,
        "detectedStack": detected_stack,
        "findings": findings,
    }


@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    if not req.captures:
        return {"answer": "No captures in your Grimoire yet. Add some captures first!", "sources": []}

    context = "\n\n".join([
        f"[{i+1}] Title: {c.get('title', '')}\nContent: {c.get('preview', '')}"
        for i, c in enumerate(req.captures[:8])
    ])

    response = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Grimoire AI, a knowledge assistant for app builders and vibe coders. "
                    "Answer questions based on the user's captured knowledge library. "
                    "Be specific, practical, and cite which captures informed your answer using [1], [2], etc. "
                    "Keep answers concise and actionable.\n\n"
                    f"User's captured knowledge:\n\n{context}"
                ),
            },
            {"role": "user", "content": req.question},
        ],
        max_tokens=500,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": req.captures[:8],
    }


# ─── Auth routes ─────────────────────────────────────────────────────────────

async def _generate_handle(pool, name: str) -> str:
    import re
    parts = name.strip().split()
    if len(parts) >= 2:
        base = (parts[0][0] + parts[-1]).lower()
    else:
        base = parts[0].lower()
    base = re.sub(r"[^a-z0-9]", "", base)
    if not base:
        base = "user"
    # find first available: base, base1, base2, ...
    candidate = base
    n = 1
    while True:
        taken = await pool.fetchval("SELECT id FROM users WHERE handle=$1", candidate)
        if not taken:
            return candidate
        candidate = f"{base}{n}"
        n += 1


@app.post("/auth/signup")
@limiter.limit("5/minute")
async def signup(req: SignupRequest, request: Request):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    pool = await get_pool()
    existing = await pool.fetchrow("SELECT id, email_verified FROM users WHERE email = $1", req.email.lower())
    if existing and existing["email_verified"]:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    email = req.email.lower().strip()
    name = req.name.strip()
    handle = await _generate_handle(pool, name)

    if existing:
        await pool.execute(
            "UPDATE users SET name=$1, password_hash=$2 WHERE email=$3",
            name, pw_hash, email,
        )
    else:
        await pool.execute(
            "INSERT INTO users (name, email, password_hash, handle) VALUES ($1, $2, $3, $4)",
            name, email, pw_hash, handle,
        )

    code = generate_otp()
    redis_set(otp_key(email, "verify"), code)
    redis_del(attempts_key(email, "verify"))
    send_otp_email(email, name, code, "verify")
    return {"message": "Verification code sent", "email": email}


@app.post("/auth/verify-otp")
@limiter.limit("10/minute")
async def verify_otp(req: OTPVerifyRequest, request: Request):
    email = req.email.lower().strip()
    key = otp_key(email, "verify")
    att_key = attempts_key(email, "verify")

    attempts = int(redis_get(att_key) or 0)
    if attempts >= OTP_MAX_ATTEMPTS:
        redis_del(key)
        raise HTTPException(status_code=429, detail="Too many attempts. Request a new code.")

    stored = redis_get(key)
    if not stored:
        raise HTTPException(status_code=400, detail="Code expired or not found. Request a new one.")

    if stored != req.code.strip():
        redis_incr(att_key)
        requests.post(f"{REDIS_URL}/expire/{att_key}/{OTP_TTL}", headers=_redis_headers(), timeout=5)
        remaining = OTP_MAX_ATTEMPTS - attempts - 1
        raise HTTPException(status_code=400, detail=f"Wrong code. {remaining} attempt{'s' if remaining != 1 else ''} left.")

    redis_del(key)
    redis_del(att_key)
    pool = await get_pool()
    user = await pool.fetchrow(
        "UPDATE users SET email_verified=true WHERE email=$1 RETURNING id, name, email",
        email,
    )
    if not user:
        raise HTTPException(status_code=404, detail="Account not found")

    token = create_token(str(user["id"]))
    return {"token": token, "user": {"id": str(user["id"]), "name": user["name"], "email": user["email"]}}


@app.post("/auth/resend-otp")
@limiter.limit("3/minute")
async def resend_otp(req: ForgotPasswordRequest, request: Request):
    email = req.email.lower().strip()
    pool = await get_pool()
    user = await pool.fetchrow("SELECT name FROM users WHERE email=$1", email)
    if not user:
        return {"message": "If that email exists, a code was sent"}
    code = generate_otp()
    redis_set(otp_key(email, "verify"), code)
    redis_del(attempts_key(email, "verify"))
    send_otp_email(email, user["name"], code, "verify")
    return {"message": "New code sent"}


@app.post("/auth/signin")
@limiter.limit("10/minute")
async def signin(req: SigninRequest, request: Request):
    pool = await get_pool()
    user = await pool.fetchrow(
        "SELECT id, name, email, password_hash, email_verified FROM users WHERE email=$1",
        req.email.lower().strip(),
    )
    if not user or not user["password_hash"]:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user["email_verified"]:
        code = generate_otp()
        redis_set(otp_key(req.email.lower(), "verify"), code)
        redis_del(attempts_key(req.email.lower(), "verify"))
        send_otp_email(user["email"], user["name"], code, "verify")
        return {"unverified": True, "email": user["email"], "message": "Please verify your email first"}

    token = create_token(str(user["id"]))
    return {"token": token, "user": {"id": str(user["id"]), "name": user["name"], "email": user["email"]}}


@app.post("/auth/forgot-password")
@limiter.limit("3/minute")
async def forgot_password(req: ForgotPasswordRequest, request: Request):
    email = req.email.lower().strip()
    pool = await get_pool()
    user = await pool.fetchrow("SELECT name FROM users WHERE email=$1 AND email_verified=true", email)
    if user:
        code = generate_otp()
        redis_set(otp_key(email, "reset"), code)
        redis_del(attempts_key(email, "reset"))
        send_otp_email(email, user["name"], code, "reset")
    return {"message": "If that email has an account, a reset code was sent"}


@app.post("/auth/reset-password")
@limiter.limit("5/minute")
async def reset_password(req: ResetPasswordRequest, request: Request):
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    email = req.email.lower().strip()
    key = otp_key(email, "reset")
    att_key = attempts_key(email, "reset")

    attempts = int(redis_get(att_key) or 0)
    if attempts >= OTP_MAX_ATTEMPTS:
        redis_del(key)
        raise HTTPException(status_code=429, detail="Too many attempts. Request a new code.")

    stored = redis_get(key)
    if not stored or stored != req.code.strip():
        redis_incr(att_key)
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    redis_del(key)
    redis_del(att_key)
    pw_hash = bcrypt.hashpw(req.new_password.encode(), bcrypt.gensalt()).decode()
    pool = await get_pool()
    user = await pool.fetchrow(
        "UPDATE users SET password_hash=$1 WHERE email=$2 RETURNING id, name, email",
        pw_hash, email,
    )
    if not user:
        raise HTTPException(status_code=404, detail="Account not found")
    token = create_token(str(user["id"]))
    return {"token": token, "user": {"id": str(user["id"]), "name": user["name"], "email": user["email"]}}


@app.get("/auth/me")
async def me(uid: str = Depends(current_user_id)):
    pool = await get_pool()
    user = await pool.fetchrow(
        """SELECT id, name, email, handle, bio, github_username, avatar_url,
                  creator_mode, youtube_url, twitter_url, newsletter_url, website_url,
                  follower_count, following_count
           FROM users WHERE id=$1""",
        uid,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_dict(user)


# ─── Creator Mode ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def run_migrations():
    pool = await get_pool()
    await pool.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS handle VARCHAR(50) UNIQUE;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS bio TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS creator_mode BOOLEAN DEFAULT FALSE;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS youtube_url TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS twitter_url TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS newsletter_url TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS website_url TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS creator_enabled_at TIMESTAMP;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS follower_count INTEGER DEFAULT 0;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS following_count INTEGER DEFAULT 0;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS push_token TEXT;
        CREATE TABLE IF NOT EXISTS follows (
            id SERIAL PRIMARY KEY,
            follower_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            following_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(follower_id, following_id)
        );
        CREATE INDEX IF NOT EXISTS idx_follows_follower ON follows(follower_id);
        CREATE INDEX IF NOT EXISTS idx_follows_following ON follows(following_id);
        CREATE TABLE IF NOT EXISTS threads (
            id SERIAL PRIMARY KEY,
            author_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            tags TEXT[] DEFAULT '{}',
            upvotes INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            is_resolved BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS thread_replies (
            id SERIAL PRIMARY KEY,
            thread_id INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
            author_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            body TEXT NOT NULL,
            upvotes INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS thread_votes (
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            thread_id INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
            vote INTEGER NOT NULL,
            PRIMARY KEY(user_id, thread_id)
        );
        CREATE TABLE IF NOT EXISTS reply_votes (
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            reply_id INTEGER NOT NULL REFERENCES thread_replies(id) ON DELETE CASCADE,
            PRIMARY KEY(user_id, reply_id)
        );
        CREATE INDEX IF NOT EXISTS idx_threads_created ON threads(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_replies_thread ON thread_replies(thread_id, created_at);
        ALTER TABLE users ADD COLUMN IF NOT EXISTS reputation_points INTEGER DEFAULT 0;
        CREATE TABLE IF NOT EXISTS creator_applications (
            id SERIAL PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            status TEXT DEFAULT 'pending',
            motivation TEXT NOT NULL,
            sample_content TEXT NOT NULL,
            rejection_reason TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            reviewed_at TIMESTAMP,
            UNIQUE(user_id)
        );
        CREATE TABLE IF NOT EXISTS packets (
            id SERIAL PRIMARY KEY,
            author_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT 'founder',
            cover_emoji TEXT DEFAULT '📦',
            status TEXT DEFAULT 'draft',
            rejection_reason TEXT,
            total_reads INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS packet_chapters (
            id SERIAL PRIMARY KEY,
            packet_id INTEGER NOT NULL REFERENCES packets(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            chapter_order INTEGER DEFAULT 0,
            is_preview BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS packet_reads (
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            packet_id INTEGER NOT NULL REFERENCES packets(id) ON DELETE CASCADE,
            read_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY(user_id, packet_id)
        );
        CREATE INDEX IF NOT EXISTS idx_packets_status ON packets(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_chapters_packet ON packet_chapters(packet_id, chapter_order);
    """)


def _user_dict(user) -> dict:
    return {
        "id": str(user["id"]),
        "name": user["name"],
        "email": user["email"],
        "handle": user["handle"],
        "bio": user["bio"],
        "githubUsername": user.get("github_username"),
        "avatarUrl": user.get("avatar_url"),
        "creatorMode": user["creator_mode"],
        "youtubeUrl": user["youtube_url"],
        "twitterUrl": user["twitter_url"],
        "newsletterUrl": user["newsletter_url"],
        "websiteUrl": user["website_url"],
        "followerCount": user["follower_count"],
        "followingCount": user["following_count"],
    }


async def optional_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str | None:
    if not creds:
        return None
    return decode_token(creds.credentials)


HANDLE_RE = re.compile(r'^[a-z0-9_-]{3,30}$')


def _uid(uid: str) -> _uuid.UUID:
    """Convert JWT sub string to UUID for DB queries (users.id is UUID type)."""
    return _uuid.UUID(uid)


class UpdateProfileRequest(BaseModel):
    name: str | None = None
    bio: str | None = None
    handle: str | None = None
    youtube_url: str | None = None
    twitter_url: str | None = None
    newsletter_url: str | None = None
    website_url: str | None = None


class EnableCreatorRequest(BaseModel):
    handle: str
    bio: str | None = None
    youtube_url: str | None = None
    twitter_url: str | None = None
    newsletter_url: str | None = None
    website_url: str | None = None


@app.put("/auth/profile")
async def update_profile(req: UpdateProfileRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    if req.handle is not None:
        h = req.handle.lower().strip()
        if not HANDLE_RE.match(h):
            raise HTTPException(status_code=400, detail="Handle must be 3-30 chars, letters/numbers/underscore/hyphen only")
        taken = await pool.fetchval("SELECT id FROM users WHERE handle=$1 AND id!=$2", h, _uid(uid))
        if taken:
            raise HTTPException(status_code=409, detail="That handle is already taken")
        req.handle = h
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update")
    cols = ", ".join(f"{k}=${i+2}" for i, k in enumerate(updates))
    vals = list(updates.values())
    user = await pool.fetchrow(
        f"""UPDATE users SET {cols}
            WHERE id=$1
            RETURNING id, name, email, handle, bio, github_username, avatar_url,
                      creator_mode, youtube_url, twitter_url, newsletter_url, website_url,
                      follower_count, following_count""",
        _uid(uid), *vals,
    )
    return _user_dict(user)


@app.post("/creator/enable")
async def enable_creator(req: EnableCreatorRequest, uid: str = Depends(current_user_id)):
    h = req.handle.lower().strip()
    if not HANDLE_RE.match(h):
        raise HTTPException(status_code=400, detail="Handle must be 3-30 chars, letters/numbers/underscore/hyphen only")
    pool = await get_pool()
    taken = await pool.fetchval("SELECT id FROM users WHERE handle=$1 AND id!=$2", h, _uid(uid))
    if taken:
        raise HTTPException(status_code=409, detail="That handle is already taken")
    user = await pool.fetchrow(
        """UPDATE users
           SET handle=$2, bio=$3, youtube_url=$4, twitter_url=$5,
               newsletter_url=$6, website_url=$7,
               creator_mode=TRUE, creator_enabled_at=NOW()
           WHERE id=$1
           RETURNING id, name, email, handle, bio, github_username, avatar_url,
                     creator_mode, youtube_url, twitter_url, newsletter_url, website_url,
                     follower_count, following_count""",
        _uid(uid), h, req.bio, req.youtube_url, req.twitter_url, req.newsletter_url, req.website_url,
    )
    return _user_dict(user)


@app.get("/creator/{handle}")
async def get_creator_profile(handle: str, viewer_id: str | None = Depends(optional_user_id)):
    pool = await get_pool()
    user = await pool.fetchrow(
        """SELECT id, name, email, handle, bio, avatar_url,
                  creator_mode, youtube_url, twitter_url, newsletter_url, website_url,
                  follower_count, following_count, creator_enabled_at
           FROM users WHERE handle=$1 AND creator_mode=TRUE""",
        handle.lower(),
    )
    if not user:
        raise HTTPException(status_code=404, detail="Creator not found")
    result = {
        "id": str(user["id"]),
        "name": user["name"],
        "handle": user["handle"],
        "bio": user["bio"],
        "avatarUrl": user["avatar_url"],
        "youtubeUrl": user["youtube_url"],
        "twitterUrl": user["twitter_url"],
        "newsletterUrl": user["newsletter_url"],
        "websiteUrl": user["website_url"],
        "followerCount": user["follower_count"],
        "followingCount": user["following_count"],
        "creatorSince": user["creator_enabled_at"].isoformat() if user["creator_enabled_at"] else None,
        "isFollowing": False,
    }
    if viewer_id:
        is_following = await pool.fetchval(
            "SELECT 1 FROM follows WHERE follower_id=$1 AND following_id=$2",
            int(viewer_id), int(user["id"]),
        )
        result["isFollowing"] = bool(is_following)
    return result


@app.post("/users/{user_id}/follow")
async def follow_user(user_id: int, uid: str = Depends(current_user_id)):
    if str(user_id) == uid:
        raise HTTPException(status_code=400, detail="You can't follow yourself")
    pool = await get_pool()
    target = await pool.fetchval("SELECT id FROM users WHERE id=$1 AND creator_mode=TRUE", user_id)
    if not target:
        raise HTTPException(status_code=404, detail="Creator not found")
    try:
        await pool.execute("INSERT INTO follows (follower_id, following_id) VALUES ($1, $2)", _uid(uid), user_id)
        await pool.execute("UPDATE users SET follower_count = follower_count + 1 WHERE id=$1", user_id)
        await pool.execute("UPDATE users SET following_count = following_count + 1 WHERE id=$1", _uid(uid))
    except asyncpg.UniqueViolationError:
        pass  # already following
    return {"following": True}


@app.delete("/users/{user_id}/follow")
async def unfollow_user(user_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    deleted = await pool.execute(
        "DELETE FROM follows WHERE follower_id=$1 AND following_id=$2", _uid(uid), user_id
    )
    if deleted != "DELETE 0":
        await pool.execute("UPDATE users SET follower_count = GREATEST(0, follower_count - 1) WHERE id=$1", user_id)
        await pool.execute("UPDATE users SET following_count = GREATEST(0, following_count - 1) WHERE id=$1", _uid(uid))
    return {"following": False}


@app.get("/creator/{handle}/followers")
async def get_creator_followers(handle: str, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    creator = await pool.fetchrow("SELECT id FROM users WHERE handle=$1 AND creator_mode=TRUE", handle.lower())
    if not creator:
        raise HTTPException(status_code=404, detail="Creator not found")
    rows = await pool.fetch(
        """SELECT u.id, u.name, u.handle, u.avatar_url, u.creator_mode
           FROM follows f JOIN users u ON u.id = f.follower_id
           WHERE f.following_id=$1 ORDER BY f.created_at DESC LIMIT 50""",
        creator["id"],
    )
    return [{"id": str(r["id"]), "name": r["name"], "handle": r["handle"],
             "avatarUrl": r["avatar_url"], "creatorMode": r["creator_mode"]} for r in rows]


@app.get("/creator/{handle}/following")
async def get_creator_following(handle: str, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    creator = await pool.fetchrow("SELECT id FROM users WHERE handle=$1", handle.lower())
    if not creator:
        raise HTTPException(status_code=404, detail="User not found")
    rows = await pool.fetch(
        """SELECT u.id, u.name, u.handle, u.avatar_url, u.creator_mode
           FROM follows f JOIN users u ON u.id = f.following_id
           WHERE f.follower_id=$1 ORDER BY f.created_at DESC LIMIT 50""",
        creator["id"],
    )
    return [{"id": str(r["id"]), "name": r["name"], "handle": r["handle"],
             "avatarUrl": r["avatar_url"], "creatorMode": r["creator_mode"]} for r in rows]


# ─── Threads ──────────────────────────────────────────────────────────────────

class CreateThreadRequest(BaseModel):
    title: str
    body: str
    tags: list[str] = []

class UpdateThreadRequest(BaseModel):
    title: str | None = None
    body: str | None = None
    tags: list[str] | None = None

class CreateReplyRequest(BaseModel):
    body: str

class UpdateReplyRequest(BaseModel):
    body: str

class PushTokenRequest(BaseModel):
    token: str


def _send_push(token: str | None, title: str, body: str):
    if not token or not token.startswith("ExponentPushToken"):
        return
    try:
        requests.post(
            "https://exp.host/--/api/v2/push/send",
            json={"to": token, "title": title, "body": body, "sound": "default"},
            timeout=5,
        )
    except Exception:
        pass


def _thread_row(t, my_vote: int | None = None) -> dict:
    return {
        "id": str(t["id"]),
        "title": t["title"],
        "body": t["body"],
        "tags": list(t["tags"]) if t["tags"] else [],
        "authorId": str(t["author_id"]),
        "authorName": t["author_name"],
        "authorHandle": t["author_handle"],
        "createdAt": t["created_at"].isoformat(),
        "updatedAt": t["updated_at"].isoformat() if t.get("updated_at") else None,
        "upvotes": t["upvotes"],
        "replyCount": t["reply_count"],
        "isResolved": t["is_resolved"],
        "myVote": my_vote,
    }


def _reply_row(r, my_voted: bool = False, is_author: bool = False) -> dict:
    return {
        "id": str(r["id"]),
        "threadId": str(r["thread_id"]),
        "body": r["body"],
        "authorId": str(r["author_id"]),
        "authorName": r["author_name"],
        "authorHandle": r["author_handle"],
        "createdAt": r["created_at"].isoformat(),
        "updatedAt": r["updated_at"].isoformat() if r.get("updated_at") else None,
        "upvotes": r["upvotes"],
        "myVote": my_voted,
        "isAuthor": is_author,
    }


@app.put("/auth/push-token")
async def update_push_token(req: PushTokenRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    await pool.execute("UPDATE users SET push_token=$1 WHERE id=$2", req.token, _uid(uid))
    return {"ok": True}


@app.get("/threads")
async def list_threads(
    tag: str | None = None,
    offset: int = 0,
    limit: int = 30,
    uid: str | None = Depends(optional_user_id),
):
    pool = await get_pool()
    where = "WHERE t.updated_at IS NULL OR TRUE"
    args: list = []
    if tag:
        where = "WHERE $1 = ANY(t.tags)"
        args.append(tag)

    rows = await pool.fetch(
        f"""SELECT t.id, t.title, t.body, t.tags, t.author_id, t.upvotes,
                   t.reply_count, t.is_resolved, t.created_at, t.updated_at,
                   u.name AS author_name, u.handle AS author_handle
            FROM threads t JOIN users u ON u.id = t.author_id
            {where}
            ORDER BY t.created_at DESC
            LIMIT ${len(args)+1} OFFSET ${len(args)+2}""",
        *args, limit, offset,
    )

    my_votes: dict[int, int] = {}
    if uid and rows:
        thread_ids = [r["id"] for r in rows]
        vote_rows = await pool.fetch(
            "SELECT thread_id, vote FROM thread_votes WHERE user_id=$1 AND thread_id=ANY($2)",
            _uid(uid), thread_ids,
        )
        my_votes = {v["thread_id"]: v["vote"] for v in vote_rows}

    return [_thread_row(r, my_votes.get(r["id"])) for r in rows]


@app.post("/threads")
async def create_thread(req: CreateThreadRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    if not req.title.strip() or not req.body.strip():
        raise HTTPException(status_code=400, detail="Title and body required")
    tags = [t.lower().strip() for t in req.tags[:5] if t.strip()]
    row = await pool.fetchrow(
        """INSERT INTO threads (author_id, title, body, tags)
           VALUES ($1, $2, $3, $4)
           RETURNING id, title, body, tags, author_id, upvotes, reply_count, is_resolved, created_at, updated_at""",
        _uid(uid), req.title.strip(), req.body.strip(), tags,
    )
    user = await pool.fetchrow("SELECT name, handle FROM users WHERE id=$1", _uid(uid))
    result = dict(row)
    result["author_name"] = user["name"]
    result["author_handle"] = user["handle"]
    return _thread_row(result)


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: int, uid: str | None = Depends(optional_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT t.id, t.title, t.body, t.tags, t.author_id, t.upvotes,
                  t.reply_count, t.is_resolved, t.created_at, t.updated_at,
                  u.name AS author_name, u.handle AS author_handle
           FROM threads t JOIN users u ON u.id = t.author_id
           WHERE t.id=$1""",
        thread_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")

    replies = await pool.fetch(
        """SELECT r.id, r.thread_id, r.body, r.author_id, r.upvotes,
                  r.created_at, r.updated_at,
                  u.name AS author_name, u.handle AS author_handle
           FROM thread_replies r JOIN users u ON u.id = r.author_id
           WHERE r.thread_id=$1
           ORDER BY r.created_at ASC""",
        thread_id,
    )

    my_thread_vote: int | None = None
    my_reply_votes: set[int] = set()
    if uid:
        tv = await pool.fetchrow(
            "SELECT vote FROM thread_votes WHERE user_id=$1 AND thread_id=$2",
            _uid(uid), thread_id,
        )
        if tv:
            my_thread_vote = tv["vote"]
        if replies:
            rv = await pool.fetch(
                "SELECT reply_id FROM reply_votes WHERE user_id=$1 AND reply_id=ANY($2)",
                _uid(uid), [r["id"] for r in replies],
            )
            my_reply_votes = {v["reply_id"] for v in rv}

    return {
        **_thread_row(row, my_thread_vote),
        "replies": [
            _reply_row(r, r["id"] in my_reply_votes, uid is not None and str(r["author_id"]) == uid)
            for r in replies
        ],
    }


@app.put("/threads/{thread_id}")
async def update_thread(thread_id: int, req: UpdateThreadRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id FROM threads WHERE id=$1", thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your thread")
    updates, args = [], [thread_id]
    if req.title is not None:
        args.append(req.title.strip()); updates.append(f"title=${len(args)}")
    if req.body is not None:
        args.append(req.body.strip()); updates.append(f"body=${len(args)}")
    if req.tags is not None:
        args.append([t.lower().strip() for t in req.tags[:5]]); updates.append(f"tags=${len(args)}")
    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update")
    args.append(datetime.utcnow()); updates.append(f"updated_at=${len(args)}")
    await pool.execute(f"UPDATE threads SET {', '.join(updates)} WHERE id=$1", *args)
    return {"ok": True}


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id FROM threads WHERE id=$1", thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your thread")
    await pool.execute("DELETE FROM threads WHERE id=$1", thread_id)
    return {"ok": True}


@app.post("/threads/{thread_id}/vote")
async def vote_thread(thread_id: int, vote: int, uid: str = Depends(current_user_id)):
    if vote not in (1, -1):
        raise HTTPException(status_code=400, detail="vote must be 1 or -1")
    pool = await get_pool()
    existing = await pool.fetchrow(
        "SELECT vote FROM thread_votes WHERE user_id=$1 AND thread_id=$2",
        _uid(uid), thread_id,
    )
    if existing:
        if existing["vote"] == vote:
            await pool.execute("DELETE FROM thread_votes WHERE user_id=$1 AND thread_id=$2", _uid(uid), thread_id)
            await pool.execute("UPDATE threads SET upvotes=GREATEST(0,upvotes-$1) WHERE id=$2", vote, thread_id)
            return {"myVote": None}
        else:
            await pool.execute("UPDATE thread_votes SET vote=$1 WHERE user_id=$2 AND thread_id=$3", vote, _uid(uid), thread_id)
            await pool.execute("UPDATE threads SET upvotes=upvotes+$1 WHERE id=$2", vote * 2, thread_id)
    else:
        await pool.execute("INSERT INTO thread_votes(user_id,thread_id,vote) VALUES($1,$2,$3)", _uid(uid), thread_id, vote)
        await pool.execute("UPDATE threads SET upvotes=upvotes+$1 WHERE id=$2", vote, thread_id)
    new_votes = await pool.fetchval("SELECT upvotes FROM threads WHERE id=$1", thread_id)
    return {"myVote": vote, "upvotes": new_votes}


@app.post("/threads/{thread_id}/resolve")
async def resolve_thread_api(thread_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id FROM threads WHERE id=$1", thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your thread")
    await pool.execute("UPDATE threads SET is_resolved=TRUE WHERE id=$1", thread_id)
    return {"ok": True}


@app.post("/threads/{thread_id}/replies")
async def create_reply(thread_id: int, req: CreateReplyRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    if not req.body.strip():
        raise HTTPException(status_code=400, detail="Body required")
    thread = await pool.fetchrow("SELECT author_id, title, is_resolved FROM threads WHERE id=$1", thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread["is_resolved"]:
        raise HTTPException(status_code=400, detail="Thread is resolved")

    user = await pool.fetchrow("SELECT name, handle, push_token FROM users WHERE id=$1", _uid(uid))
    row = await pool.fetchrow(
        """INSERT INTO thread_replies(thread_id, author_id, body)
           VALUES($1,$2,$3)
           RETURNING id, thread_id, body, author_id, upvotes, created_at, updated_at""",
        thread_id, _uid(uid), req.body.strip(),
    )
    await pool.execute("UPDATE threads SET reply_count=reply_count+1 WHERE id=$1", thread_id)

    # Push notification to thread author (not self)
    if str(thread["author_id"]) != uid:
        author = await pool.fetchrow("SELECT push_token FROM users WHERE id=$1", thread["author_id"])
        if author and author["push_token"]:
            _send_push(
                author["push_token"],
                f"{user['name']} replied",
                thread["title"],
            )

    result = dict(row)
    result["author_name"] = user["name"]
    result["author_handle"] = user["handle"]
    return _reply_row(result, False, True)


@app.put("/threads/{thread_id}/replies/{reply_id}")
async def update_reply(thread_id: int, reply_id: int, req: UpdateReplyRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id FROM thread_replies WHERE id=$1 AND thread_id=$2", reply_id, thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="Reply not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your reply")
    await pool.execute(
        "UPDATE thread_replies SET body=$1, updated_at=$2 WHERE id=$3",
        req.body.strip(), datetime.utcnow(), reply_id,
    )
    return {"ok": True}


@app.delete("/threads/{thread_id}/replies/{reply_id}")
async def delete_reply(thread_id: int, reply_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id FROM thread_replies WHERE id=$1 AND thread_id=$2", reply_id, thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="Reply not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your reply")
    await pool.execute("DELETE FROM thread_replies WHERE id=$1", reply_id)
    await pool.execute("UPDATE threads SET reply_count=GREATEST(0,reply_count-1) WHERE id=$1", thread_id)
    return {"ok": True}


@app.post("/threads/{thread_id}/replies/{reply_id}/vote")
async def vote_reply(thread_id: int, reply_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    existing = await pool.fetchrow(
        "SELECT 1 FROM reply_votes WHERE user_id=$1 AND reply_id=$2", _uid(uid), reply_id,
    )
    if existing:
        await pool.execute("DELETE FROM reply_votes WHERE user_id=$1 AND reply_id=$2", _uid(uid), reply_id)
        await pool.execute("UPDATE thread_replies SET upvotes=GREATEST(0,upvotes-1) WHERE id=$1", reply_id)
        new_votes = await pool.fetchval("SELECT upvotes FROM thread_replies WHERE id=$1", reply_id)
        return {"myVote": False, "upvotes": new_votes}
    else:
        await pool.execute("INSERT INTO reply_votes(user_id,reply_id) VALUES($1,$2)", _uid(uid), reply_id)
        await pool.execute("UPDATE thread_replies SET upvotes=upvotes+1 WHERE id=$1", reply_id)
        new_votes = await pool.fetchval("SELECT upvotes FROM thread_replies WHERE id=$1", reply_id)
        return {"myVote": True, "upvotes": new_votes}


# ═══════════════════════════════════════════════════════════════════════════════
# REPUTATION SYNC
# ═══════════════════════════════════════════════════════════════════════════════

@app.put("/auth/reputation")
async def sync_reputation(req: ReputationSyncRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    await pool.execute("UPDATE users SET reputation_points=$1 WHERE id=$2", req.points, _uid(uid))
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════════════
# CREATOR APPLICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/creator/apply")
async def creator_apply(req: CreatorApplicationRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    user = await pool.fetchrow("SELECT reputation_points, creator_mode FROM users WHERE id=$1", _uid(uid))
    if user["creator_mode"]:
        raise HTTPException(status_code=400, detail="Already a creator")

    # Auto-approve if user has Expert level (500+ Gears)
    auto_approve = (user["reputation_points"] or 0) >= 500
    status = "approved" if auto_approve else "pending"
    reviewed_at = "NOW()" if auto_approve else "NULL"

    existing = await pool.fetchrow("SELECT id, status FROM creator_applications WHERE user_id=$1", _uid(uid))
    if existing:
        if existing["status"] == "approved":
            raise HTTPException(status_code=400, detail="Application already approved")
        await pool.execute(
            "UPDATE creator_applications SET status=$1, motivation=$2, sample_content=$3, reviewed_at=CASE WHEN $1='approved' THEN NOW() ELSE NULL END WHERE user_id=$4",
            status, req.motivation, req.sample_content, _uid(uid),
        )
    else:
        await pool.execute(
            "INSERT INTO creator_applications(user_id, status, motivation, sample_content, reviewed_at) VALUES($1,$2,$3,$4,CASE WHEN $2='approved' THEN NOW() ELSE NULL END)",
            _uid(uid), status, req.motivation, req.sample_content,
        )

    if auto_approve:
        await pool.execute("UPDATE users SET creator_mode=TRUE, creator_enabled_at=NOW() WHERE id=$1", _uid(uid))

    return {"status": status, "autoApproved": auto_approve}


@app.get("/creator/application")
async def get_creator_application(uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, status, rejection_reason, created_at, reviewed_at FROM creator_applications WHERE user_id=$1",
        _uid(uid),
    )
    if not row:
        return {"status": "none"}
    return {
        "id": str(row["id"]),
        "status": row["status"],
        "rejectionReason": row["rejection_reason"],
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
        "reviewedAt": row["reviewed_at"].isoformat() if row["reviewed_at"] else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PACKET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _chapter_row(ch) -> dict:
    return {
        "id": str(ch["id"]),
        "packetId": str(ch["packet_id"]),
        "title": ch["title"],
        "content": ch["content"],
        "chapterOrder": ch["chapter_order"],
        "isPreview": ch["is_preview"],
    }


def _packet_row(p, chapters=None, is_subscriber=False) -> dict:
    out = {
        "id": str(p["id"]),
        "authorId": str(p["author_id"]),
        "authorName": p.get("author_name") or "Creator",
        "authorHandle": p.get("author_handle"),
        "title": p["title"],
        "description": p["description"],
        "category": p["category"],
        "coverEmoji": p["cover_emoji"],
        "status": p["status"],
        "totalReads": p["total_reads"],
        "chapterCount": p.get("chapter_count", 0),
        "createdAt": p["created_at"].isoformat() if p.get("created_at") else None,
        "updatedAt": p["updated_at"].isoformat() if p.get("updated_at") else None,
    }
    if chapters is not None:
        visible = [_chapter_row(c) for c in chapters if c["is_preview"] or is_subscriber]
        out["chapters"] = visible
        out["previewChapterCount"] = sum(1 for c in chapters if c["is_preview"])
    return out


async def _require_creator(uid: str, pool) -> None:
    user = await pool.fetchrow("SELECT creator_mode FROM users WHERE id=$1", _uid(uid))
    if not user or not user["creator_mode"]:
        raise HTTPException(status_code=403, detail="Creator access required. Apply at /creator/apply")


# ═══════════════════════════════════════════════════════════════════════════════
# PACKET ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/packets")
async def list_packets(
    category: str | None = None,
    offset: int = 0,
    limit: int = 20,
    uid: str | None = Depends(lambda creds=Depends(bearer_scheme): decode_token(creds.credentials) if creds else None),
):
    pool = await get_pool()
    params: list = ["published"]
    sql = """
        SELECT p.*, u.name AS author_name, u.handle AS author_handle,
               (SELECT COUNT(*) FROM packet_chapters WHERE packet_id=p.id) AS chapter_count
        FROM packets p JOIN users u ON u.id=p.author_id
        WHERE p.status=$1
    """
    if category:
        params.append(category)
        sql += f" AND p.category=${len(params)}"
    params += [limit, offset]
    sql += f" ORDER BY p.total_reads DESC, p.created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}"
    rows = await pool.fetch(sql, *params)
    return [_packet_row(r) for r in rows]


@app.get("/my/packets")
async def my_packets(uid: str = Depends(current_user_id)):
    pool = await get_pool()
    await _require_creator(uid, pool)
    rows = await pool.fetch(
        """SELECT p.*,
               (SELECT COUNT(*) FROM packet_chapters WHERE packet_id=p.id) AS chapter_count
           FROM packets p WHERE p.author_id=$1 ORDER BY p.created_at DESC""",
        _uid(uid),
    )
    return [_packet_row(r) for r in rows]


@app.post("/packets")
async def create_packet(req: CreatePacketRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    await _require_creator(uid, pool)
    row = await pool.fetchrow(
        "INSERT INTO packets(author_id,title,description,category,cover_emoji) VALUES($1,$2,$3,$4,$5) RETURNING *",
        _uid(uid), req.title.strip(), req.description.strip(), req.category, req.cover_emoji,
    )
    return _packet_row(row)


@app.get("/packets/{packet_id}")
async def get_packet(
    packet_id: int,
    uid: str | None = Depends(lambda creds=Depends(bearer_scheme): decode_token(creds.credentials) if creds else None),
):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT p.*, u.name AS author_name, u.handle AS author_handle FROM packets p JOIN users u ON u.id=p.author_id WHERE p.id=$1",
        packet_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Packet not found")

    is_author = uid and str(row["author_id"]) == uid
    if row["status"] != "published" and not is_author:
        raise HTTPException(status_code=404, detail="Packet not found")

    chapters = await pool.fetch(
        "SELECT * FROM packet_chapters WHERE packet_id=$1 ORDER BY chapter_order, id",
        packet_id,
    )
    # Subscribers and authors see all chapters; others see preview only
    is_subscriber = bool(uid)  # For now, any authenticated user can read — paywall added later
    return _packet_row(row, chapters=chapters, is_subscriber=is_subscriber or is_author)


@app.put("/packets/{packet_id}")
async def update_packet(packet_id: int, req: UpdatePacketRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id, status FROM packets WHERE id=$1", packet_id)
    if not row:
        raise HTTPException(status_code=404, detail="Packet not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your packet")
    if row["status"] not in ("draft", "rejected"):
        raise HTTPException(status_code=400, detail="Can only edit draft or rejected packets")
    updates, params = [], [packet_id]
    if req.title is not None:
        params.append(req.title.strip()); updates.append(f"title=${len(params)}")
    if req.description is not None:
        params.append(req.description.strip()); updates.append(f"description=${len(params)}")
    if req.category is not None:
        params.append(req.category); updates.append(f"category=${len(params)}")
    if req.cover_emoji is not None:
        params.append(req.cover_emoji); updates.append(f"cover_emoji=${len(params)}")
    if updates:
        params.append(datetime.now(timezone.utc))
        await pool.execute(
            f"UPDATE packets SET {', '.join(updates)}, updated_at=${len(params)} WHERE id=$1",
            *params,
        )
    return {"ok": True}


@app.delete("/packets/{packet_id}")
async def delete_packet(packet_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id FROM packets WHERE id=$1", packet_id)
    if not row:
        raise HTTPException(status_code=404, detail="Packet not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your packet")
    await pool.execute("DELETE FROM packets WHERE id=$1", packet_id)
    return {"ok": True}


@app.post("/packets/{packet_id}/submit")
async def submit_packet(packet_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id, status FROM packets WHERE id=$1", packet_id)
    if not row:
        raise HTTPException(status_code=404, detail="Packet not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your packet")
    if row["status"] not in ("draft", "rejected"):
        raise HTTPException(status_code=400, detail="Only draft or rejected packets can be submitted")
    chapter_count = await pool.fetchval("SELECT COUNT(*) FROM packet_chapters WHERE packet_id=$1", packet_id)
    if chapter_count == 0:
        raise HTTPException(status_code=400, detail="Add at least one chapter before submitting")
    await pool.execute(
        "UPDATE packets SET status='pending_review', updated_at=NOW() WHERE id=$1", packet_id,
    )
    return {"ok": True, "status": "pending_review"}


@app.post("/packets/{packet_id}/read")
async def record_read(packet_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT status FROM packets WHERE id=$1", packet_id)
    if not row or row["status"] != "published":
        raise HTTPException(status_code=404, detail="Packet not found")
    existing = await pool.fetchrow(
        "SELECT 1 FROM packet_reads WHERE user_id=$1 AND packet_id=$2", _uid(uid), packet_id,
    )
    if not existing:
        await pool.execute(
            "INSERT INTO packet_reads(user_id,packet_id) VALUES($1,$2)", _uid(uid), packet_id,
        )
        await pool.execute("UPDATE packets SET total_reads=total_reads+1 WHERE id=$1", packet_id)
    return {"ok": True}


# ─── Chapters ────────────────────────────────────────────────────────────────

@app.post("/packets/{packet_id}/chapters")
async def add_chapter(packet_id: int, req: CreateChapterRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT author_id, status FROM packets WHERE id=$1", packet_id)
    if not row:
        raise HTTPException(status_code=404, detail="Packet not found")
    if str(row["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your packet")
    if row["status"] not in ("draft", "rejected"):
        raise HTTPException(status_code=400, detail="Can only edit draft or rejected packets")
    ch = await pool.fetchrow(
        "INSERT INTO packet_chapters(packet_id,title,content,chapter_order,is_preview) VALUES($1,$2,$3,$4,$5) RETURNING *",
        packet_id, req.title.strip(), req.content.strip(), req.chapter_order, req.is_preview,
    )
    return _chapter_row(ch)


@app.put("/packets/{packet_id}/chapters/{chapter_id}")
async def update_chapter(packet_id: int, chapter_id: int, req: UpdateChapterRequest, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    p = await pool.fetchrow("SELECT author_id FROM packets WHERE id=$1", packet_id)
    if not p or str(p["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your packet")
    updates, params = [], [chapter_id, packet_id]
    if req.title is not None:
        params.append(req.title.strip()); updates.append(f"title=${len(params)}")
    if req.content is not None:
        params.append(req.content.strip()); updates.append(f"content=${len(params)}")
    if req.chapter_order is not None:
        params.append(req.chapter_order); updates.append(f"chapter_order=${len(params)}")
    if req.is_preview is not None:
        params.append(req.is_preview); updates.append(f"is_preview=${len(params)}")
    if updates:
        await pool.execute(
            f"UPDATE packet_chapters SET {', '.join(updates)} WHERE id=$1 AND packet_id=$2",
            *params,
        )
    return {"ok": True}


@app.delete("/packets/{packet_id}/chapters/{chapter_id}")
async def delete_chapter(packet_id: int, chapter_id: int, uid: str = Depends(current_user_id)):
    pool = await get_pool()
    p = await pool.fetchrow("SELECT author_id FROM packets WHERE id=$1", packet_id)
    if not p or str(p["author_id"]) != uid:
        raise HTTPException(status_code=403, detail="Not your packet")
    await pool.execute("DELETE FROM packet_chapters WHERE id=$1 AND packet_id=$2", chapter_id, packet_id)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/applications/{app_id}/review")
async def admin_review_application(app_id: int, req: AdminReviewRequest, request: Request):
    require_admin(request)
    pool = await get_pool()
    row = await pool.fetchrow("SELECT user_id FROM creator_applications WHERE id=$1", app_id)
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    if req.action == "approve":
        await pool.execute(
            "UPDATE creator_applications SET status='approved', reviewed_at=NOW() WHERE id=$1", app_id,
        )
        await pool.execute(
            "UPDATE users SET creator_mode=TRUE, creator_enabled_at=NOW() WHERE id=$1", row["user_id"],
        )
    elif req.action == "reject":
        await pool.execute(
            "UPDATE creator_applications SET status='rejected', rejection_reason=$1, reviewed_at=NOW() WHERE id=$2",
            req.reason, app_id,
        )
    else:
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")
    return {"ok": True}


@app.post("/admin/packets/{packet_id}/review")
async def admin_review_packet(packet_id: int, req: AdminReviewRequest, request: Request):
    require_admin(request)
    pool = await get_pool()
    row = await pool.fetchrow("SELECT id FROM packets WHERE id=$1", packet_id)
    if not row:
        raise HTTPException(status_code=404, detail="Packet not found")
    if req.action == "approve":
        await pool.execute(
            "UPDATE packets SET status='published', updated_at=NOW() WHERE id=$1", packet_id,
        )
    elif req.action == "reject":
        await pool.execute(
            "UPDATE packets SET status='rejected', rejection_reason=$1, updated_at=NOW() WHERE id=$2",
            req.reason, packet_id,
        )
    else:
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")
    return {"ok": True}


@app.get("/admin/applications")
async def admin_list_applications(request: Request, status: str = "pending"):
    require_admin(request)
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT a.*, u.name AS user_name, u.email AS user_email, u.reputation_points
           FROM creator_applications a JOIN users u ON u.id=a.user_id
           WHERE a.status=$1 ORDER BY a.created_at""",
        status,
    )
    return [{
        "id": str(r["id"]),
        "userId": str(r["user_id"]),
        "userName": r["user_name"],
        "userEmail": r["user_email"],
        "reputationPoints": r["reputation_points"],
        "status": r["status"],
        "motivation": r["motivation"],
        "sampleContent": r["sample_content"],
        "rejectionReason": r["rejection_reason"],
        "createdAt": r["created_at"].isoformat() if r["created_at"] else None,
    } for r in rows]


@app.get("/admin/packets")
async def admin_list_packets(request: Request, status: str = "pending_review"):
    require_admin(request)
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT p.*, u.name AS author_name, u.email AS author_email,
               (SELECT COUNT(*) FROM packet_chapters WHERE packet_id=p.id) AS chapter_count
           FROM packets p JOIN users u ON u.id=p.author_id
           WHERE p.status=$1 ORDER BY p.created_at""",
        status,
    )
    return [_packet_row(r) for r in rows]
