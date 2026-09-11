import os
import io
import json
import re
import uuid
import urllib.parse
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv

# Load .env BEFORE importing model_client so Mistral configuration is available.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

import markdown
import numpy as np
import requests
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    current_app,
    g,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# local model client & embeddings
from model_client import (
    chat as model_chat,
    generate_embeddings,
    embed_query,
    get_youtube_recommendations,
    MistralRateLimitError,
    MistralAPIError,
)

# pdf libs
import pypdf

# PyMuPDF — optional (not in requirements, graceful fallback)
try:
    import fitz
except ImportError:
    fitz = None

from PIL import Image



def _mistral_error_payload(exc):
    """Return a safe, presentation-friendly API error payload."""
    if isinstance(exc, MistralRateLimitError):
        return {
            "status": "error",
            "error": "Mistral API rate limit reached. Please wait a moment and try again.",
            "code": 429,
        }
    return {
        "status": "error",
        "error": str(exc),
    }


def get_youtube_metadata(url):
    try:
        r = requests.get(
            f"https://www.youtube.com/oembed?url={url}&format=json",
            timeout=5,
        )
        data = r.json()
        return {
            "title": data.get("title", ""),
            "author": data.get("author_name", ""),
        }
    except Exception:
        return {"title": "", "author": ""}


# ── Basic config ──────────────────────────────────────────────────────────────
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
EXAMPLE_PATH = os.getenv("EXAMPLE_FILE", "")

SESSION_COOKIE_NAME = "rag_session_id"

# ── Flask + DB ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")

database_url = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR}/rag_study_buddy.db"
)

# Railway / PostgreSQL compatibility
if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql://",
        1
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Stable secret key so sessions do not randomly change after restart
app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "eduquery-demo-secret-key-2026"
)
# Keep uploads reasonable for a student-demo deployment.
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
db = SQLAlchemy(app)

# ── OCR — optional ────────────────────────────────────────────────────────────
try:
    import easyocr
    OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
    _ocr_reader = easyocr.Reader(OCR_LANGS, gpu=False)
except Exception:
    _ocr_reader = None

# ── YouTube transcript — optional ─────────────────────────────────────────────
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
except Exception:
    YouTubeTranscriptApi = None

    class TranscriptsDisabled(Exception):
        pass

    class NoTranscriptFound(Exception):
        pass


# ── DB MODELS ─────────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = "user"
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(255), unique=True, nullable=False)
    name          = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


class UserSession(db.Model):
    __tablename__ = "user_session"
    id         = db.Column(db.String(64), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)


class UploadedDocument(db.Model):
    __tablename__   = "uploaded_document"
    id              = db.Column(db.Integer, primary_key=True)
    session_id      = db.Column(db.String(64), db.ForeignKey("user_session.id"))
    stored_filename = db.Column(db.String(255), nullable=False)
    original_name   = db.Column(db.String(255), nullable=False)
    uploaded_at     = db.Column(db.DateTime, default=datetime.utcnow)


class ChatMessage(db.Model):
    __tablename__ = "chat_message"
    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), db.ForeignKey("user_session.id"))
    role       = db.Column(db.String(10), nullable=False)
    source     = db.Column(db.String(30), nullable=False)
    text       = db.Column(db.Text, nullable=False)
    filename   = db.Column(db.String(255), nullable=True)
    video_id   = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Flashcard(db.Model):
    __tablename__   = "flashcard"
    id              = db.Column(db.Integer, primary_key=True)
    session_id      = db.Column(db.String(64), db.ForeignKey("user_session.id"))
    front           = db.Column(db.Text, nullable=False)
    back            = db.Column(db.Text, nullable=False)
    source_filename = db.Column(db.String(255), nullable=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed        = db.Column(db.Boolean, default=False)
    difficulty      = db.Column(db.String(10), nullable=True)



# ── EMPLOYEE PERSONALIZATION MODELS ───────────────────────────────────────────

class EmployeeProfile(db.Model):
    __tablename__ = "employee_profile"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)
    designation = db.Column(db.String(150), default="")
    department = db.Column(db.String(150), default="")
    organization = db.Column(db.String(150), default="")
    work_domain = db.Column(db.String(150), default="")
    current_assignment = db.Column(db.Text, default="")
    experience_years = db.Column(db.Float, default=0)
    education = db.Column(db.String(250), default="")
    preferred_language = db.Column(db.String(30), default="English")
    learning_goal = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Competency(db.Model):
    __tablename__ = "competency"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, default="")


class EmployeeCompetency(db.Model):
    __tablename__ = "employee_competency"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    competency_id = db.Column(db.Integer, db.ForeignKey("competency.id"), nullable=False)
    current_level = db.Column(db.Float, default=0)
    required_level = db.Column(db.Float, default=3)
    last_assessed = db.Column(db.DateTime, nullable=True)
    competency = db.relationship("Competency", backref="employee_competencies")


class Course(db.Model):
    __tablename__ = "course"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(250), nullable=False)
    provider = db.Column(db.String(100), default="iGOT")
    category = db.Column(db.String(100), default="")
    competency = db.Column(db.String(150), default="")
    level = db.Column(db.String(50), default="Beginner")
    description = db.Column(db.Text, default="")
    duration_hours = db.Column(db.Float, default=2)
    url = db.Column(db.String(500), default="#")
    active = db.Column(db.Boolean, default=True)


class LearningProgress(db.Model):
    __tablename__ = "learning_progress"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    progress = db.Column(db.Float, default=0)
    last_score = db.Column(db.Float, nullable=True)
    completed = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    course = db.relationship("Course", backref="learning_progress")


class AssessmentResult(db.Model):
    __tablename__ = "assessment_result"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    competency_id = db.Column(db.Integer, db.ForeignKey("competency.id"), nullable=False)
    score = db.Column(db.Float, default=0)
    total_questions = db.Column(db.Integer, default=0)
    assessed_level = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    competency = db.relationship("Competency", backref="assessment_results")


# ── FIX: Create tables at module load so gunicorn workers find them ───────────
# Previously db.create_all() was only inside `if __name__ == "__main__"`
# which gunicorn NEVER executes → tables never created → OperationalError.
def seed_competencies():
    data = [
        ("Survey Design", "Statistical", "Design and methodology of statistical surveys."),
        ("Sampling Methods", "Statistical", "Sampling techniques and statistical sampling design."),
        ("Official Statistics", "Statistical", "Concepts and practices related to official statistics."),
        ("Data Quality", "Statistical", "Statistical data validation, quality and metadata."),
        ("Labour Statistics", "Statistical", "Concepts and methods used in labour statistics."),
        ("Time Series Analysis", "Statistical", "Analysis of time-dependent data."),
        ("Python", "Technical", "Python programming for statistical and data analysis."),
        ("SQL", "Technical", "Database querying and statistical data extraction."),
        ("R Programming", "Technical", "R programming for statistics and data analysis."),
        ("Data Visualization", "Technical", "Visualization and communication of statistical data."),
        ("GIS", "Technical", "Geospatial information systems and spatial analysis."),
        ("Artificial Intelligence", "Technical", "AI concepts and applications in official statistics."),
        ("Machine Learning", "Technical", "Machine learning methods for statistical applications."),
        ("Big Data Analytics", "Technical", "Processing and analysis of large datasets."),
        ("Cloud Computing", "Technical", "Cloud technologies and scalable data processing."),
        ("Cybersecurity", "Digital Governance", "Cybersecurity principles for government systems."),
        ("Data Privacy", "Digital Governance", "Data protection and privacy principles."),
        ("Digital Governance", "Digital Governance", "Digital transformation and governance concepts."),
        ("API Integration", "Digital Governance", "Design and consumption of secure APIs."),
        ("Leadership", "Behavioral", "Leadership and team management."),
        ("Communication", "Behavioral", "Professional communication and presentation."),
        ("Decision Making", "Behavioral", "Evidence-based decision making."),
    ]
    for name, category, description in data:
        if not Competency.query.filter_by(name=name).first():
            db.session.add(Competency(name=name, category=category, description=description))


def seed_courses():
    data = [
        ("SQL for Statistical Data Analysis", "iGOT", "Technical", "SQL", "Intermediate", "SQL concepts for querying and analysing statistical datasets.", 8),
        ("Python for Data Analysis", "iGOT", "Technical", "Python", "Beginner", "Python fundamentals for statistical and data analysis.", 10),
        ("GIS Fundamentals", "iGOT", "Technical", "GIS", "Beginner", "Introduction to GIS, spatial data and geographical analysis.", 8),
        ("Artificial Intelligence for Government", "iGOT", "Technical", "Artificial Intelligence", "Beginner", "Introduction to AI applications in government.", 6),
        ("Machine Learning for Data Analysis", "NSSTA", "Technical", "Machine Learning", "Intermediate", "Machine learning methods for statistical data analysis.", 12),
        ("Data Visualization for Official Statistics", "NSSTA", "Technical", "Data Visualization", "Intermediate", "Visual communication of official statistical information.", 6),
        ("Survey Sampling Methods", "NSSTA", "Statistical", "Sampling Methods", "Intermediate", "Sampling methods used in official statistical surveys.", 10),
    ]
    for title, provider, category, competency, level, description, hours in data:
        if not Course.query.filter_by(title=title).first():
            db.session.add(Course(title=title, provider=provider, category=category, competency=competency, level=level, description=description, duration_hours=hours))




with app.app_context():
    db.create_all()
    seed_competencies()
    seed_courses()
    db.session.commit()
# ── SESSION HELPERS ───────────────────────────────────────────────────────────

@app.before_request
def load_or_create_session():
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    new = False
    if not sid:
        sid = uuid.uuid4().hex
        new = True

    g.session_id = sid
    g.new_session = new
    g.user = None

    # Wrapped in try/except — defensive guard in case DB has a transient issue
    try:
        us = db.session.get(UserSession, sid)
        if not us:
            us = UserSession(id=sid)
            db.session.add(us)
            db.session.commit()
        if us.user_id:
            g.user = db.session.get(User, us.user_id)
    except Exception as e:
        current_app.logger.error("Session load error: %s", e)
        db.session.rollback()


@app.after_request
def set_session_cookie(response):
    try:
        # OLD: only sets on brand new sessions
        # if getattr(g, "new_session", False):

        # NEW: set whenever we have a session id (refreshes on login too)
        if getattr(g, "session_id", None):
            response.set_cookie(
                SESSION_COOKIE_NAME,
                g.session_id,
                max_age=60 * 60 * 24 * 30,
                httponly=True,
                samesite="Lax",
            )
    except RuntimeError:
        pass
    return response


def _save_chat(
    session_id: str,
    role: str,
    source: str,
    text: str,
    filename: Optional[str] = None,
    video_id: Optional[str] = None,
):
    try:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            source=source,
            text=text,
            filename=filename,
            video_id=video_id,
        )
        db.session.add(msg)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        current_app.logger.warning("Failed to save chat message: %s", e)


# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.route("/auth", methods=["GET", "POST"])
def auth_page():
    if request.method == "GET":
        ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
        return render_template("auth.html", example_name=ex_name, active="auth")

    mode     = request.form.get("mode", "login")
    email    = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "").strip()
    name     = (request.form.get("name") or "").strip() or "Guest"
    sid      = getattr(g, "session_id", None)
    error    = None

    if mode == "signup":
        if not email or not password:
            error = "Email and password required."
        elif User.query.filter_by(email=email).first():
            error = "User already exists."
        else:
            user = User(
                email=email,
                name=name,
                password_hash=generate_password_hash(password),
            )
            db.session.add(user)
            db.session.commit()
            if sid:
                us = db.session.get(UserSession, sid)
                if us:
                    us.user_id = user.id
                    db.session.commit()
            return redirect(url_for("dashboard"))

    elif mode == "login":
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            error = "Invalid email or password."
        else:
            if sid:
                us = db.session.get(UserSession, sid)
                if us:
                    us.user_id = user.id
                    db.session.commit()
            return redirect(url_for("dashboard"))

    elif mode == "guest":
        return redirect(url_for("dashboard"))

    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template("auth.html", example_name=ex_name, active="auth", error=error)


@app.route("/logout")
def logout():
    sid = getattr(g, "session_id", None)
    if sid:
        us = db.session.get(UserSession, sid)
        if us:
            us.user_id = None
            db.session.commit()
    resp = redirect(url_for("landing"))
    resp.set_cookie(SESSION_COOKIE_NAME, "", expires=0)
    return resp


# ── PDF / IMAGE / OCR HELPERS ─────────────────────────────────────────────────

def extract_text_from_pdf(path: str):
    try:
        reader = pypdf.PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return pages
    except Exception:
        return []


def extract_images_from_pdf(path: str, max_images_per_page: int = 3):
    images = []
    if fitz is None:
        return images
    try:
        doc = fitz.open(path)
    except Exception:
        return images
    for i in range(len(doc)):
        page = doc[i]
        img_list = page.get_images(full=True)
        count = 0
        for im in img_list:
            if count >= max_images_per_page:
                break
            xref = im[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
                count += 1
            except Exception:
                continue
    return images


def ocr_image_local(img: Image.Image) -> str:
    if _ocr_reader is None:
        return ""
    try:
        arr = np.array(img)
        texts = _ocr_reader.readtext(arr, detail=0)
        return "\n".join(texts)
    except Exception:
        return ""


def chunk_texts(pages, chunk_size: int = 1000, chunk_overlap: int = 200):
    chunks = []
    for p in pages:
        s = p.strip()
        if not s:
            continue
        i = 0
        while i < len(s):
            chunks.append(s[i: i + chunk_size])
            i += chunk_size - chunk_overlap
    return chunks


def build_chunks_for_file(path: str, max_images_ocr: int = 2):
    ext = Path(path).suffix.lower()
    pages = []
    if ext == ".pdf":
        pages = extract_text_from_pdf(path)
        imgs = extract_images_from_pdf(path)
        for img in imgs[:max_images_ocr]:
            t = ocr_image_local(img)
            if t and t.strip():
                pages.append(t)
    elif ext in {".txt", ".md"}:
        try:
            with open(path, "r", encoding="utf-8") as f:
                pages = [f.read()]
        except Exception:
            pages = []
    else:
        try:
            img = Image.open(path).convert("RGB")
            t = ocr_image_local(img)
            if t:
                pages.append(t)
        except Exception:
            pages = []
    return chunk_texts(pages)


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

def similarity_search(chunks, query, top_k: int = 6):
    chunk_embs = generate_embeddings(chunks)
    q_emb = embed_query(query)
    vectors = np.array(chunk_embs)
    q = np.array(q_emb)
    norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(q) + 1e-12)
    sims = vectors.dot(q) / norms
    idx = np.argsort(-sims)[:top_k]
    return [chunks[i] for i in idx]


# ── QUESTION GENERATION ───────────────────────────────────────────────────────

def build_instruction(mode: str, num_q: int, include_answers: bool) -> str:
    if mode == "mcq":
        ans_part = ',"answer":"A"' if include_answers else ''
        return (
            f"Create {num_q} multiple-choice questions (A-D) from the context. "
            f'Return JSON array: [{{"question":"...","choices":{{"A":"...","B":"...","C":"...","D":"..."}}{ans_part}}}].'
        )
    elif mode == "short":
        inst = f"Create {num_q} short-answer questions (1-3 sentences)."
        if include_answers:
            inst += " Include 'answer' field with brief answers."
        return inst + ' Return JSON array: [{"question":"...","answer":"..."}].'
    else:
        inst = f"Create {num_q} long-answer questions (detailed)."
        if include_answers:
            inst += " Include 'answer' field with comprehensive answers."
        return inst + ' Return JSON array: [{"question":"...","answer":"..."}].'

def build_custom_question_prompt(
    requirement: str,
    q_types: list,
    difficulty: str,
    num_q: int,
    language: str,
    context: str = ""
) -> str:
    ctx_block = f"\n\nDocument context:\n{context}" if context else ""
    return (
        f"You are an expert question paper generator.\n"
        f"Generate exactly {num_q} questions based on the user's requirement below.\n"
        f"Question types to use: {', '.join(q_types)}\n"
        f"Difficulty: {difficulty}\n"
        f"Language: {language}\n"
        f"Distribute questions intelligently across the requested types.\n"
        f"{ctx_block}\n\n"
        f"User requirement: {requirement}\n\n"
        f"Return ONLY a JSON array, no backticks, no explanation:\n"
        f'[{{"type":"MCQ"|"Short Answer"|"Long Answer"|"True/False",'
        f'"difficulty":"Easy"|"Medium"|"Hard",'
        f'"question":"...",'
        f'"options":["A...","B...","C...","D..."],'  # only for MCQ
        f'"answer":"..."}}]'
    )

# ── YOUTUBE HELPERS ───────────────────────────────────────────────────────────

YOUTUBE_REGEX = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=[\w\-]+(&\S*)?|youtu\.be/[\w\-]+)",
    re.IGNORECASE,
)


def extract_youtube_url(text: str) -> Optional[str]:
    if not text:
        return None
    m = YOUTUBE_REGEX.search(text)
    return m.group(0) if m else None


def extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    if "youtu.be/" in url:
        return url.rstrip("/").split("/")[-1].split("?")[0]
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    return qs["v"][0] if "v" in qs and qs["v"] else None


def language_to_iso(lang: str) -> List[str]:
    mapping = {
        "english": ["en"], "hindi": ["hi", "en"], "hinglish": ["hi", "en"],
        "spanish": ["es", "en"], "french": ["fr", "en"], "german": ["de", "en"],
        "portuguese": ["pt", "en"], "bengali": ["bn", "en"], "tamil": ["ta", "en"],
        "telugu": ["te", "en"], "marathi": ["mr", "en"],
        "gujarati": ["gu", "en"], "urdu": ["ur", "en"],
    }
    return mapping.get((lang or "").strip().lower(), ["en"])


def get_youtube_transcript_text(video_id: str, language: str = "English") -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi as YTA
    except ImportError as e:
        raise RuntimeError("youtube-transcript-api not installed") from e
    langs = language_to_iso(language)
    try:
        transcript = YTA.get_transcript(video_id, languages=langs)
    except Exception:
        try:
            transcript = YTA.get_transcript(video_id, languages=["en"])
        except Exception as e:
            raise RuntimeError(f"No transcript available: {e}") from e
    return " ".join(seg.get("text", "") for seg in transcript if seg.get("text"))


# ── IMPORTANT QUESTION SHORTLISTING ──────────────────────────────────────────

def shortlist_important_questions(raw_text: str, num_q: int = 20, language: str = "English"):
    prompt = f"""
You are an expert exam paper analyst.
Shortlist the TOP {num_q} most important exam questions from the text below.
Prefer questions that are repeated, high-weightage, or cover core concepts.
Merge similar questions into one. Ignore headers, page numbers, marking schemes.

Text:
{raw_text}

Return ONLY valid JSON — no backticks, no extra text:
[{{"question": "...", "reason": "..."}}]

Language: {language}
"""
    raw = model_chat(prompt, max_tokens=1000, temperature=0.3)
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            try:
                return json.loads(m.group(1).replace("'", '"'))
            except Exception:
                return {"raw": raw}
    return {"raw": raw}

@app.route("/generate_custom_questions", methods=["POST"])
def generate_custom_questions():
    requirement = (request.form.get("requirement") or "").strip()
    filename    = (request.form.get("filename") or "").strip()
    q_types     = request.form.getlist("q_types") or ["MCQ", "Short Answer"]
    difficulty  = (request.form.get("difficulty") or "Mixed").strip()
    language    = (request.form.get("language") or "English").strip()
    sid         = getattr(g, "session_id", None)

    try:
        num_q = max(1, min(int(request.form.get("num_questions", "5")), 20))
    except ValueError:
        num_q = 5

    if not requirement:
        return jsonify({"status": "error", "error": "requirement is required"}), 400

    context = ""
    if filename:
        path = UPLOAD_DIR / filename
        if not path.exists():
            return jsonify({"status": "error", "error": "File not found"}), 404
        chunks = build_chunks_for_file(str(path))
        if chunks:
            context = "\n\n---\n\n".join(chunks[:8])

    prompt = build_custom_question_prompt(
        requirement, q_types, difficulty, num_q, language, context
    )

    try:
        raw = model_chat(prompt, max_tokens=1000, temperature=0.4)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return jsonify({"status": "error", "error": "Model did not return valid JSON", "raw": raw}), 500

    try:
        questions = json.loads(m.group(0))
    except Exception:
        try:
            questions = json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return jsonify({"status": "error", "error": "JSON parse failed", "raw": raw}), 500

    if sid:
        _save_chat(sid, "user", "custom_questions", requirement, filename or None)

    return jsonify({
        "status": "ok",
        "language": language,
        "num_questions": len(questions),
        "questions": questions
    }), 200

# ── FILE CHAT HANDLER ─────────────────────────────────────────────────────────

def _handle_chat_for_path(resolved_path: str, question: str, language: str):
    if not resolved_path or not question:
        return 400, {"status": "error", "error": "missing parameters"}
    if not os.path.isabs(resolved_path):
        resolved_path = str(UPLOAD_DIR / resolved_path)
    if not os.path.exists(resolved_path):
        return 404, {"status": "error", "error": f"file not found: {resolved_path}"}
    try:
        chunks = build_chunks_for_file(str(resolved_path))
        if not chunks:
            return 400, {"status": "error", "error": "no extractable text from file"}
        top_ctx = similarity_search(chunks, question, top_k=6)
        context = "\n\n---\n\n".join(top_ctx)
        prompt = (
            f"You are an assistant for question answering over documents.\n"
            f"Use ONLY the given context to answer. Answer in {language}.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )
        raw = model_chat(prompt)
        return 200, {"status": "ok", "answer": raw}
    except MistralRateLimitError as e:
        return 429, _mistral_error_payload(e)
    except Exception as e:
        current_app.logger.exception("Error in chat handler")
        return 500, {"status": "error", "error": str(e)}


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/study_result", methods=["POST"])
def study_result_page():
    youtube_url = (request.form.get("youtube_url") or "").strip()
    notes       = (request.form.get("notes") or "").strip()
    language    = (request.form.get("language") or "English").strip()
    sid         = getattr(g, "session_id", None)

    if not youtube_url and not notes:
        return redirect(url_for("study_page"))

    yt_url = extract_youtube_url(youtube_url) if youtube_url else None

    if yt_url:
        meta = get_youtube_metadata(yt_url)
        title = meta.get("title", "") or "Educational video"
        author = meta.get("author", "")
        vid = extract_youtube_id(yt_url)

        # Prefer the real transcript when available so the answer is grounded
        # in the video rather than relying only on its title.
        transcript_text = ""
        if vid:
            try:
                transcript_text = get_youtube_transcript_text(vid, language=language)
            except Exception:
                transcript_text = ""

        transcript_context = ""
        if transcript_text:
            transcript_chunks = chunk_texts([transcript_text], 1200, 200)
            transcript_context = "\n\n---\n\n".join(transcript_chunks[:6])

        prompt = (
            "You are an expert teacher and YouTube content analyst.\n"
            "Explain the educational video clearly and accurately.\n\n"
            f"Video title: {title}\nCreator: {author}\n"
            f"Student request: {notes or 'Explain the main concepts and important points.'}\n"
            + (f"Transcript context:\n{transcript_context}\n\n" if transcript_context else
               "No transcript was available. Use only the title and student request; "
               "do not invent specific claims about unseen video content.\n\n")
            + f"Answer in {language}.\n"
            "Structure: Topic Overview, Detailed Explanation, Key Concepts, "
            "Examples, Important Points, and 3 Exam Questions."
        )

        try:
            answer = model_chat(prompt, max_tokens=800, temperature=0.4)
        except MistralRateLimitError as e:
            return jsonify(_mistral_error_payload(e)), 429
        except MistralAPIError as e:
            return jsonify(_mistral_error_payload(e)), getattr(e, "status_code", 502)
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

        html_answer = markdown.markdown(answer, extensions=["fenced_code", "tables"])
        recommendation_text = (
            f"Video title: {title}\nCreator: {author}\nStudent notes: {notes}\n"
            f"Transcript: {transcript_text[:3000]}"
        )
        recommendations = get_youtube_recommendations(recommendation_text)
        return jsonify({
            "status": "ok",
            "answer": html_answer,
            "recommendations": recommendations,
            "video_id": vid,
            "grounded_by_transcript": bool(transcript_text),
        })
    try:
        answer = model_chat(
            f"You are a helpful study assistant. Explain clearly in {language}.\n\nNotes:\n{notes}",
            max_tokens=800, temperature=0.3
        )
    except MistralRateLimitError as e:
        return jsonify(_mistral_error_payload(e)), 429
    except MistralAPIError as e:
        return jsonify(_mistral_error_payload(e)), getattr(e, "status_code", 502)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    if sid:
        _save_chat(sid, "ai", "study_chat", answer)
    recommendations = get_youtube_recommendations(notes)
    return jsonify({"status": "ok", "language": language, "answer": answer, "recommendations": recommendations}), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("app_home"))
    f   = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return "Not allowed file type", 400
    uid   = uuid.uuid4().hex
    saved = UPLOAD_DIR / f"{uid}{ext}"
    f.save(saved)
    sid = getattr(g, "session_id", None)
    if sid:
        try:
            db.session.add(UploadedDocument(
                session_id=sid,
                stored_filename=saved.name,
                original_name=f.filename or saved.name,
            ))
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.warning("Failed to save upload row: %s", e)
    return redirect(url_for("process_file", filename=saved.name))


@app.route("/process/<filename>", methods=["GET", "POST"])
def process_file(filename):
    path = UPLOAD_DIR / filename
    if not path.exists():
        return "Not found", 404
    if request.method == "GET":
        ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
        return render_template("ask.html", filename=filename, example_name=ex_name)

    mode            = request.form.get("mode", "mcq")
    num_q           = int(request.form.get("num_questions", "4"))
    include_answers = request.form.get("include_answers") == "on"
    language        = request.form.get("language", "English")

    try:
        chunks = build_chunks_for_file(str(path))
        if not chunks:
            return "No extractable text", 400
        context = "\n\n---\n\n".join(chunks[:8])
        inst    = build_instruction(mode, num_q, include_answers)
        prompt  = f"You are a question generator. {inst}\nContext:\n{context}\nOutput language: {language}\nReturn only JSON."
        raw     = model_chat(prompt)
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if m:
            try:
                qdata = json.loads(m.group(1))
            except Exception:
                try:
                    qdata = json.loads(m.group(1).replace("'", '"'))
                except Exception:
                    qdata = {"raw": raw}
        else:
            qdata = {"raw": raw}
        ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
        return render_template(
            "results.html", filename=filename,
            questions=qdata, language=language, example_name=ex_name
        )
    except MistralRateLimitError as e:
        return "Mistral API rate limit reached. Please wait a moment and try again.", 429
    except Exception as e:
        current_app.logger.exception("Error in process_file")
        return f"Error: {e}", 500


@app.route("/chat/<filename>", methods=["POST"])
def chat_file(filename):
    question = request.form.get("question", "").strip()
    language = request.form.get("language", "English")
    sid      = getattr(g, "session_id", None)
    if sid and question:
        _save_chat(sid, "user", "doc_chat", question, filename)
    status, payload = _handle_chat_for_path(str(UPLOAD_DIR / filename), question, language)
    if sid and status == 200 and payload.get("status") == "ok":
        ans = payload.get("answer", "")
        _save_chat(sid, "ai", "doc_chat",
                   ans[:4000] if isinstance(ans, str) else json.dumps(ans)[:4000], filename)
    return jsonify(payload), status


@app.route("/chat", methods=["POST"])
def chat_generic():
    question = (request.form.get("question") or "").strip()
    language = (request.form.get("language") or "English").strip()
    file_url = request.form.get("file_url")
    filename = request.form.get("filename")
    sid      = getattr(g, "session_id", None)
    if not question:
        return jsonify({"status": "error", "error": "missing question"}), 400
    resolved_path = file_url if file_url else (str(UPLOAD_DIR / filename) if filename else None)
    if not resolved_path:
        return jsonify({"status": "error", "error": "missing filename or file_url"}), 400
    if sid:
        _save_chat(sid, "user", "doc_chat_generic", question, filename)
    status, payload = _handle_chat_for_path(resolved_path, question, language)
    if sid and status == 200 and payload.get("status") == "ok":
        ans = payload.get("answer", "")
        _save_chat(sid, "ai", "doc_chat_generic",
                   ans[:4000] if isinstance(ans, str) else json.dumps(ans)[:4000], filename)
    return jsonify(payload), status


@app.route("/study_chat", methods=["POST"])
def study_chat():
    if request.is_json:
        data     = request.get_json(silent=True) or {}
        message  = (data.get("message") or "").strip()
        language = (data.get("language") or "English").strip()
    else:
        message  = (request.form.get("message") or "").strip()
        language = (request.form.get("language") or "English").strip()

    if not message:
        return jsonify({"status": "error", "error": "missing message"}), 400

    sid    = getattr(g, "session_id", None)
    yt_url = extract_youtube_url(message)

    if yt_url:
        vid             = extract_youtube_id(yt_url)
        transcript_text = ""
        if vid:
            try:
                transcript_text = get_youtube_transcript_text(vid, language=language)
            except Exception:
                pass

        meta   = get_youtube_metadata(yt_url)
        title  = meta.get("title", "")
        author = meta.get("author", "")

        if transcript_text:
            transcript_chunks = chunk_texts([transcript_text], 1200, 200)
            context = "\n\n---\n\n".join(transcript_chunks[:6])
            mode = "transcript"
        else:
            context = ""
            mode = "inferred_context"

        prompt = (
            f"You are an expert teacher and YouTube content analyst.\n"
            f"Explain the video clearly to a student.\n\n"
            f"Video: {title} by {author}\nURL: {yt_url}\n"
            + (f"Transcript: {context}\n" if context else "")
            + f"Mode: {mode}\n\nStudent: {message}\n\n"
            f"DO NOT mention missing transcript or say you cannot access the video.\n"
            f"Structure: 1. Topic Overview 2. Detailed Explanation 3. Key Concepts "
            f"4. Real-world Examples 5. Important Points 6. 3-5 Exam Questions with Answers\n"
            f"Language: {language}"
        )
        try:
            answer = model_chat(prompt, max_tokens=800, temperature=0.4)
        except MistralRateLimitError as e:
            return jsonify(_mistral_error_payload(e)), 429
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

        if sid:
            _save_chat(sid, "user", "youtube_chat", message, video_id=vid)
            _save_chat(sid, "ai",   "youtube_chat", answer,  video_id=vid)

        return jsonify({
            "status": "ok", "mode": "youtube",
            "youtube_url": yt_url, "video_id": vid,
            "title": title, "author": author,
            "language": language, "answer": answer,
        }), 200

    # Normal chat
    prompt = f"You are a helpful study assistant.\n\nStudent message:\n{message}\n\nExplain clearly in {language}."
    try:
        answer = model_chat(prompt, max_tokens=1000, temperature=0.3)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    if sid:
        _save_chat(sid, "user", "study_chat", message)
        _save_chat(sid, "ai",   "study_chat", answer)

    return jsonify({"status": "ok", "mode": "chat", "language": language, "answer": answer}), 200


@app.route("/generate_flashcards", methods=["POST"])
def generate_flashcards():
    filename = (request.form.get("filename") or "").strip()
    raw_text = (request.form.get("text") or "").strip()
    language = (request.form.get("language") or "English").strip()
    sid      = getattr(g, "session_id", None)
    try:
        num_cards = max(1, min(int(request.form.get("num_cards", "10")), 30))
    except ValueError:
        num_cards = 10

    if filename:
        path = UPLOAD_DIR / filename
        if not path.exists():
            return jsonify({"status": "error", "error": "File not found"}), 404
        chunks = build_chunks_for_file(str(path))
        if not chunks:
            return jsonify({"status": "error", "error": "No text extracted"}), 400
        context = "\n\n".join(chunks[:10])
    elif raw_text:
        context = raw_text
    else:
        return jsonify({"status": "error", "error": "Provide filename or text"}), 400

    prompt = (
        f"Create {num_cards} flashcards from the content below.\n"
        f"Each has FRONT (question/term) and BACK (answer/definition). Be concise and exam-focused.\n\n"
        f"Content:\n{context}\n\n"
        f"Return ONLY a JSON array — no backticks:\n"
        f'[{{"front": "...", "back": "..."}}]\n\nLanguage: {language}'
    )
    try:
        raw = model_chat(prompt, max_tokens=800, temperature=0.3)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return jsonify({"status": "error", "error": "Model did not return valid JSON", "raw": raw}), 500
    try:
        cards_data = json.loads(m.group(0))
    except Exception:
        try:
            cards_data = json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return jsonify({"status": "error", "error": "JSON parse failed", "raw": raw}), 500

    saved_cards = []
    if sid and isinstance(cards_data, list):
        for card in cards_data:
            if not isinstance(card, dict):
                continue
            front = card.get("front", "").strip()
            back  = card.get("back", "").strip()
            if front and back:
                db.session.add(Flashcard(
                    session_id=sid, front=front, back=back,
                    source_filename=filename or None,
                ))
                saved_cards.append({"front": front, "back": back})
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.warning("Flashcard save error: %s", e)

    return jsonify({
        "status": "ok", "language": language,
        "num_cards": len(saved_cards), "flashcards": saved_cards
    }), 200


@app.route("/flashcards", methods=["GET"])
def flashcards_page():
    sid   = getattr(g, "session_id", None)
    cards = (
        Flashcard.query.filter_by(session_id=sid)
        .order_by(Flashcard.created_at.desc()).all()
    ) if sid else []
    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template("flashcards.html", cards=cards, example_name=ex_name, active="flashcards")


@app.route("/flashcard/<int:card_id>/review", methods=["POST"])
def review_flashcard(card_id):
    sid  = getattr(g, "session_id", None)
    card = Flashcard.query.filter_by(id=card_id, session_id=sid).first()
    if not card:
        return jsonify({"status": "error", "error": "Card not found"}), 404
    difficulty = (request.json or {}).get("difficulty", "medium")
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    card.reviewed   = True
    card.difficulty = difficulty
    db.session.commit()
    return jsonify({"status": "ok", "card_id": card_id, "difficulty": difficulty})


@app.route("/flashcard/<int:card_id>", methods=["DELETE"])
def delete_flashcard(card_id):
    sid  = getattr(g, "session_id", None)
    card = Flashcard.query.filter_by(id=card_id, session_id=sid).first()
    if not card:
        return jsonify({"status": "error", "error": "Card not found"}), 404
    db.session.delete(card)
    db.session.commit()
    return jsonify({"status": "ok"})


@app.route("/summarize", methods=["POST"])
def summarize_document():
    filename = (request.form.get("filename") or "").strip()
    raw_text = (request.form.get("text") or "").strip()
    language = (request.form.get("language") or "English").strip()
    style    = (request.form.get("style") or "brief").strip().lower()
    if style not in ("brief", "detailed"):
        style = "brief"

    if filename:
        path = UPLOAD_DIR / filename
        if not path.exists():
            return jsonify({"status": "error", "error": "File not found"}), 404
        chunks = build_chunks_for_file(str(path))
        if not chunks:
            return jsonify({"status": "error", "error": "No text extracted"}), 400
        context = "\n\n".join(chunks[:12])
    elif raw_text:
        context = raw_text
    else:
        return jsonify({"status": "error", "error": "Provide filename or text"}), 400

    detail = (
        "Give a comprehensive paragraph-by-paragraph summary covering all key topics."
        if style == "detailed"
        else "Give a concise 5-7 bullet-point summary of the most important ideas."
    )
    try:
        answer = model_chat(
            f"You are an expert study summarizer.\n\n{detail}\n\nContent:\n{context}\n\nLanguage: {language}",
            max_tokens=800, temperature=0.3
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    return jsonify({"status": "ok", "language": language, "style": style, "summary": answer}), 200


@app.route("/important_questions", methods=["POST"])
def important_questions():
    files = request.files.getlist("papers")
    if not files:
        return jsonify({"status": "error", "error": "No files uploaded"}), 400
    language = (request.form.get("language") or "English").strip()
    try:
        num_q = int(request.form.get("num_questions", "20"))
    except ValueError:
        num_q = 20

    all_chunks = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        uid        = uuid.uuid4().hex
        saved_path = UPLOAD_DIR / f"{uid}{ext}"
        f.save(saved_path)
        try:
            all_chunks.extend(build_chunks_for_file(str(saved_path)))
        except Exception as e:
            current_app.logger.warning("Error processing %s: %s", saved_path, e)

    if not all_chunks:
        return jsonify({"status": "error", "error": "Could not extract text from any paper"}), 400

    combined_text = "\n\n---\n\n".join(all_chunks[:40])
    try:
        shortlisted = shortlist_important_questions(combined_text, num_q=num_q, language=language)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    return jsonify({
        "status": "ok", "language": language,
        "requested_num_questions": num_q, "important_questions": shortlisted
    }), 200


@app.route("/important", methods=["GET"])
def important_questions_page():
    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template("important_questions.html", example_name=ex_name, active="important")


@app.route("/history", methods=["GET"])
def history():
    sid      = getattr(g, "session_id", None)
    docs     = UploadedDocument.query.filter_by(session_id=sid).order_by(UploadedDocument.uploaded_at.desc()).limit(20).all() if sid else []
    messages = ChatMessage.query.filter_by(session_id=sid).order_by(ChatMessage.created_at.desc()).limit(40).all() if sid else []
    ex_name  = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template("history.html", docs=docs, messages=messages, example_name=ex_name, active="history")





@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/")
def landing():
    return render_template("landing.html", active="home")


@app.route("/app")
def app_home():
    return render_template("dashboard.html")


def _current_user_required():
    return getattr(g, "user", None)


def calculate_gap(current, required):
    return max(0.0, float(required or 0) - float(current or 0))


def gap_priority(gap):
    if gap >= 2.5:
        return "Critical"
    if gap >= 1.5:
        return "High"
    if gap >= 0.75:
        return "Medium"
    return "Low"


def recommendation_score(gap, role_relevance=0.8, semantic_similarity=0.75, learning_history=0.5):
    gap_score = min(max(gap, 0) / 5, 1)
    return round((0.40 * gap_score + 0.30 * role_relevance + 0.20 * semantic_similarity + 0.10 * learning_history) * 100, 2)


def generate_personalized_recommendations(user_id):
    skills = EmployeeCompetency.query.filter_by(user_id=user_id).all()
    recommendations = []
    for skill in skills:
        gap = calculate_gap(skill.current_level, skill.required_level)
        if gap <= 0:
            continue
        courses = Course.query.filter_by(active=True, competency=skill.competency.name).all()
        for course in courses:
            score = recommendation_score(gap)
            recommendations.append({
                "course": course,
                "competency": skill.competency.name,
                "gap": round(gap, 2),
                "priority": gap_priority(gap),
                "score": score,
                "reason": f"Your current {skill.competency.name} competency is {skill.current_level:.1f}/5 versus a required {skill.required_level:.1f}/5 for your profile."
            })
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:6]


@app.route("/profile", methods=["GET", "POST"])
def employee_profile():
    if not _current_user_required():
        return redirect(url_for("auth_page"))
    profile = EmployeeProfile.query.filter_by(user_id=g.user.id).first()
    if request.method == "POST":
        if not profile:
            profile = EmployeeProfile(user_id=g.user.id)
            db.session.add(profile)
        profile.designation = (request.form.get("designation") or "").strip()
        profile.department = (request.form.get("department") or "").strip()
        profile.organization = (request.form.get("organization") or "").strip()
        profile.work_domain = (request.form.get("work_domain") or "").strip()
        profile.current_assignment = (request.form.get("current_assignment") or "").strip()
        profile.education = (request.form.get("education") or "").strip()
        profile.learning_goal = (request.form.get("learning_goal") or "").strip()
        profile.preferred_language = (request.form.get("preferred_language") or "English").strip()
        try:
            profile.experience_years = max(0, float(request.form.get("experience_years") or 0))
        except ValueError:
            profile.experience_years = 0
        db.session.commit()
        return redirect(url_for("dashboard"))
    return render_template("profile.html", profile=profile)


@app.route("/skills", methods=["GET", "POST"])
def employee_skills():
    if not _current_user_required():
        return redirect(url_for("auth_page"))
    competencies = Competency.query.order_by(Competency.category, Competency.name).all()
    if request.method == "POST":
        for competency in competencies:
            try:
                current_level = float(request.form.get(f"skill_{competency.id}") or 0)
            except ValueError:
                current_level = 0
            try:
                required_level = float(request.form.get(f"required_{competency.id}") or 3)
            except ValueError:
                required_level = 3
            record = EmployeeCompetency.query.filter_by(user_id=g.user.id, competency_id=competency.id).first()
            if not record:
                record = EmployeeCompetency(user_id=g.user.id, competency_id=competency.id)
                db.session.add(record)
            record.current_level = max(0, min(5, current_level))
            record.required_level = max(0, min(5, required_level))
            record.last_assessed = datetime.utcnow()
        db.session.commit()
        return redirect(url_for("dashboard"))
    existing = {x.competency_id: x for x in EmployeeCompetency.query.filter_by(user_id=g.user.id).all()}
    return render_template("skills.html", competencies=competencies, existing=existing)


@app.route("/learning")
def learning_plan():
    if not _current_user_required():
        return redirect(url_for("auth_page"))
    recommendations = generate_personalized_recommendations(g.user.id)
    progress = LearningProgress.query.filter_by(user_id=g.user.id).order_by(LearningProgress.updated_at.desc()).all()
    return render_template("learning_plan.html", recommendations=recommendations, progress=progress)

@app.route("/course/<int:course_id>/start", methods=["POST"])
def start_course(course_id):
    if not _current_user_required():
        return jsonify({"status": "error", "error": "Login required"}), 401
    course = db.session.get(Course, course_id)
    if not course:
        return jsonify({"status": "error", "error": "Course not found"}), 404
    item = LearningProgress.query.filter_by(
        user_id=g.user.id,
        course_id=course.id
    ).first()
    if not item:
        item = LearningProgress(
            user_id=g.user.id,
            course_id=course.id,
            progress=5,
            completed=False
        )
        db.session.add(item)
    else:
        item.progress = max(item.progress or 0, 5)
    db.session.commit()
    # Open assessment...
    competency = Competency.query.filter_by(name=course.competency).first()
    if not competency:
        return redirect(url_for("learning_plan"))
    return redirect(url_for("assessment", competency_id=competency.id
                           )
                   )


@app.route("/progress")
def progress_page():
    if not _current_user_required():
        return redirect(url_for("auth_page"))
    competencies = EmployeeCompetency.query.filter_by(user_id=g.user.id).all()
    progress = LearningProgress.query.filter_by(user_id=g.user.id).order_by(LearningProgress.updated_at.desc()).all()
    assessments = AssessmentResult.query.filter_by(user_id=g.user.id).order_by(AssessmentResult.created_at.desc()).all()
    return render_template("progress.html", competencies=competencies, progress=progress, assessments=assessments)


ASSESSMENT_BANK = {
    "SQL": [
        ("Which SQL clause filters rows before grouping?", ["WHERE", "HAVING", "ORDER BY", "GROUP BY"], "WHERE"),
        ("Which command retrieves data from a table?", ["SELECT", "INSERT", "UPDATE", "DELETE"], "SELECT"),
        ("Which key uniquely identifies a row?", ["Primary key", "Foreign key", "Index only", "View"], "Primary key"),
        ("Which clause sorts query results?", ["ORDER BY", "SORT", "GROUP BY", "ARRANGE"], "ORDER BY"),
        ("Which function counts rows?", ["COUNT()", "TOTAL()", "ROWS()", "NUMBER()"], "COUNT()"),
    ],
    "Python": [
        ("Which keyword defines a function?", ["def", "func", "function", "define"], "def"),
        ("Which structure stores key-value pairs?", ["List", "Tuple", "Dictionary", "Set"], "Dictionary"),
        ("Which library is widely used for tabular data?", ["Pandas", "Flask", "Requests", "Pillow"], "Pandas"),
        ("What does len() return?", ["Length/size", "Type", "Memory address", "Hash only"], "Length/size"),
        ("Which symbol starts a Python comment?", ["#", "//", "--", "/*"], "#"),
    ],
    "GIS": [
        ("What does GIS stand for?", ["Geographic Information System", "Global Image Service", "Graphical Internet System", "Geometric Index Search"], "Geographic Information System"),
        ("Which data represents points, lines and polygons?", ["Vector", "Raster", "Audio", "Tabular only"], "Vector"),
        ("Raster data is organized primarily as what?", ["Cells/pixels", "Vertices only", "Rows of text", "Edges only"], "Cells/pixels"),
        ("A coordinate reference system primarily defines what?", ["How locations are represented on Earth", "File compression", "Database password", "Image brightness"], "How locations are represented on Earth"),
        ("Which operation combines nearby features based on distance?", ["Buffer", "Compile", "Hash", "Tokenize"], "Buffer"),
    ],
    "Machine Learning": [
        ("Which task predicts a continuous value?", ["Regression", "Classification", "Clustering", "Association"], "Regression"),
        ("Which method is used to reduce overfitting?", ["Regularization", "Data leakage", "Removing validation", "Increasing noise"], "Regularization"),
        ("Which metric is common for binary classification?", ["F1-score", "MSE only", "MAE only", "RMSE only"], "F1-score"),
        ("What is a validation set used for?", ["Model selection/tuning", "Final reporting only", "Storing passwords", "Database joins"], "Model selection/tuning"),
        ("What does overfitting mean?", ["Good training fit but poor generalization", "Poor training and test fit", "No model", "Only missing data"], "Good training fit but poor generalization"),
    ],
}


@app.route("/assessment/by-name/<path:name>")
def assessment_by_name(name):
    competency = Competency.query.filter_by(name=name).first()
    if not competency:
        return redirect(url_for("dashboard"))
    return redirect(url_for("assessment", competency_id=competency.id))


@app.route("/assessment/<int:competency_id>", methods=["GET", "POST"])
def assessment(competency_id):
    if not g.user:
        return redirect(url_for("auth_page"))
    competency = db.session.get(Competency, competency_id)
    if not competency:
        return redirect(url_for("dashboard"))
    questions = ASSESSMENT_BANK.get(competency.name, [
        (f"Which statement best describes {competency.name}?", ["Core professional competency", "Unrelated activity", "A file format", "A password"], "Core professional competency"),
        (f"Why is {competency.name} relevant to an employee?", ["It supports role performance", "It is always irrelevant", "It replaces every other skill", "It is only for entertainment"], "It supports role performance"),
        (f"How should {competency.name} be improved?", ["Practice and assessment", "Never use it", "Avoid feedback", "Ignore requirements"], "Practice and assessment"),
        (f"What is a good way to measure {competency.name}?", ["Performance-based assessment", "Guessing", "No measurement", "Random selection"], "Performance-based assessment"),
        (f"What should an employee do after an assessment?", ["Review weaknesses and learn", "Ignore the result", "Delete all progress", "Stop learning"], "Review weaknesses and learn"),
    ])
    if request.method == "POST":
        correct = 0
        for i, q in enumerate(questions):
            if request.form.get(f"q{i}") == q[2]:
                correct += 1
        total = len(questions)
        score = round((correct / total) * 100, 1) if total else 0
        record = EmployeeCompetency.query.filter_by(user_id=g.user.id, competency_id=competency.id).first()
        if not record:
            record = EmployeeCompetency(user_id=g.user.id, competency_id=competency.id, current_level=0, required_level=3)
            db.session.add(record)
        # Evidence-based incremental update: a perfect assessment can add up to 0.5 level.
        improvement = round((score / 100) * 0.5, 2)
        record.current_level = min(5, round((record.current_level or 0) + improvement, 2))
        record.last_assessed = datetime.utcnow()
        db.session.add(AssessmentResult(user_id=g.user.id, competency_id=competency.id, score=score, total_questions=total, assessed_level=record.current_level))
        db.session.commit()
        return render_template("assessment_result.html", competency=competency, score=score, correct=correct, total=total, new_level=record.current_level)
    return render_template("assessment.html", competency=competency, questions=questions)


@app.route("/dashboard")
def dashboard():
    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""

    if not g.user:
        return render_template(
            "dashboard.html",
            example_name=ex_name,
            profile=None,
            competencies=[],
            skill_gaps=[],
            recommendations=[],
            progress=[],
            assessments=[],
            overall_score=0,
            completed_courses=0,
            total_courses=0,
            progress_percentage=0,
            remaining_percentage=100
        )

    # Get the logged-in user's profile
    profile = EmployeeProfile.query.filter_by(
        user_id=g.user.id
    ).first()

    # Get only this user's competencies
    competencies = EmployeeCompetency.query.filter_by(
        user_id=g.user.id
    ).all()

    # Calculate skill gaps
    skill_gaps = []

    for item in competencies:
        gap = calculate_gap(
            item.current_level,
            item.required_level
        )

        if gap > 0:
            skill_gaps.append({
                "name": item.competency.name,
                "category": item.competency.category,
                "current": item.current_level,
                "required": item.required_level,
                "gap": round(gap, 2),
                "priority": gap_priority(gap)
            })

    skill_gaps.sort(
        key=lambda x: x["gap"],
        reverse=True
    )

    # Overall competency score
    total_current = sum(
        x.current_level or 0
        for x in competencies
    )

    total_required = sum(
        x.required_level or 0
        for x in competencies
    )

    overall_score = (
        round((total_current / total_required) * 100)
        if total_required
        else 0
    )

    # Personalized recommendations
    recommendations = generate_personalized_recommendations(
        g.user.id
    )

    # Get ONLY the logged-in user's learning progress
    all_progress = LearningProgress.query.filter_by(
        user_id=g.user.id
    ).all()

    # Recent learning activity
    progress = LearningProgress.query.filter_by(
        user_id=g.user.id
    ).order_by(
        LearningProgress.updated_at.desc()
    ).limit(5).all()

    # Recent assessments
    assessments = AssessmentResult.query.filter_by(
        user_id=g.user.id
    ).order_by(
        AssessmentResult.created_at.desc()
    ).limit(5).all()

    # Number of courses belonging to this user
    total_courses = len(all_progress)

    # Number of completed courses belonging to this user
    completed_courses = sum(
        1 for item in all_progress
        if item.completed
    )

    # Calculate overall learning progress
    if total_courses > 0:
        progress_percentage = round(
            sum(
                min(max(item.progress or 0, 0), 100)
                for item in all_progress
            ) / total_courses
        )
    else:
        progress_percentage = 0

    remaining_percentage = max(
        100 - progress_percentage,
        0
    )

    return render_template(
        "dashboard.html",
        example_name=ex_name,
        profile=profile,
        competencies=competencies,
        skill_gaps=skill_gaps[:6],
        recommendations=recommendations,
        progress=progress,
        assessments=assessments,
        overall_score=overall_score,
        completed_courses=completed_courses,
        total_courses=total_courses,
        progress_percentage=progress_percentage,
        remaining_percentage=remaining_percentage
    )
def study_page():
    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template("study.html", example_name=ex_name, active="study")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.2.0"}), 200


@app.errorhandler(MistralRateLimitError)
def mistral_rate_limit_error(e):
    return jsonify(_mistral_error_payload(e)), 429


@app.errorhandler(MistralAPIError)
def mistral_api_error(e):
    return jsonify({"status": "error", "error": str(e), "code": getattr(e, "status_code", 502)}), getattr(e, "status_code", 502)


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"status": "error", "error": "File is too large. Maximum upload size is 15 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    current_app.logger.exception("INTERNAL SERVER ERROR")
    return jsonify({
        "status": "error",
        "error": str(e)
    }), 500
@app.context_processor
def inject_user():
    return dict(current_user=getattr(g, "user", None))
    # ── ENTRYPOINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run("0.0.0.0", port=port, debug=False)

