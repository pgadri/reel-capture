import os
import glob
import re
import json
import base64
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


def summarize(transcript: str, raw_title: str) -> dict:
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
                "content": f"Raw title from video: {raw_title}\n\nTranscript:\n{transcript[:8000]}",
            },
        ],
        max_tokens=900,
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

        summary = summarize(transcript, title)
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
            pass  # GitHub push is optional — capture still succeeds without it

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

    if existing:
        await pool.execute("UPDATE users SET name=$1, password_hash=$2 WHERE email=$3", name, pw_hash, email)
    else:
        await pool.execute(
            "INSERT INTO users (name, email, password_hash) VALUES ($1, $2, $3)",
            name, email, pw_hash,
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
    user = await pool.fetchrow("SELECT id, name, email, github_username, avatar_url FROM users WHERE id=$1", uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": str(user["id"]), "name": user["name"], "email": user["email"],
            "githubUsername": user["github_username"], "avatarUrl": user["avatar_url"]}
