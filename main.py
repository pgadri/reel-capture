import os
import glob
import re
import json
import base64
import tempfile
from datetime import datetime

import yt_dlp
import openai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "pgadri/my-captures")


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
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Untitled")
        uploader = info.get("uploader", "Unknown")
        return title, uploader


def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = get_groq_client().audio.transcriptions.create(model="whisper-large-v3-turbo", file=f)
    return result.text


def summarize(transcript: str, raw_title: str) -> dict:
    """Returns { title: str, bullets: list[str] } extracted by GPT."""
    response = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge extraction assistant for app builders and vibe coders. "
                    "Given a video transcript, respond with a JSON object containing exactly two keys: "
                    "'title' (a specific, concrete title under 12 words — not generic) and "
                    "'bullets' (an array of 3 to 5 actionable insights, each under 100 characters, "
                    "no bullet symbols, concrete and specific not vague)."
                ),
            },
            {
                "role": "user",
                "content": f"Raw title from video: {raw_title}\n\nTranscript:\n{transcript[:4000]}",
            },
        ],
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return {
            "title": data.get("title") or raw_title,
            "bullets": [str(b) for b in data.get("bullets", [])][:5],
        }
    except Exception:
        return {"title": raw_title, "bullets": []}


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

        try:
            note_url = push_to_github(filename, note)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GitHub push failed: {str(e)}")

        return {
            "success": True,
            "title": clean_title,
            "note_url": note_url,
            "preview": preview,
            "bullets": bullets,
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
async def analyze_image(req: AnalyzeImageRequest):
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
async def chat(req: ChatRequest):
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
