import os
import shutil
import logging
import json # For serializing list of floats for embeddings
import numpy as np # For efficient serialization of embeddings
import pickle # Alternative for serialization, be cautious with untrusted data. Numpy is generally safer.

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Boolean, Text, DateTime, LargeBinary
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base, selectinload
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# Security and Authentication
from passlib.context import CryptContext
from jose import JWTError, jwt

# PDF and NLP
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Asynchronous tasks
from celery import Celery
from celery.signals import worker_ready # To load models in worker

# --- Configuration & Setup ---
# Load .env variables (simple way for single file, consider python-dotenv for robust .env handling)
# In a real app, use a proper config management library or environment variables directly.
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key] = value

SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key_please_change")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

DATABASE_URL = "sqlite:///./esg_scorer.db"
UPLOAD_DIRECTORY = "./uploaded_reports"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)
celery_app.conf.update(
    task_track_started=True,
    # task_serializer='json', # Ensure tasks use JSON for args/kwargs
    # result_serializer='json',
    # accept_content=['json'],
)


# SQLAlchemy setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI app instance
app = FastAPI(title="ESG Scoring Assistant API")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NLP Model Loading ---
# Determine device for PyTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"PyTorch will use device: {device}")

# Global variables for models (load once)
sentence_model: Optional[SentenceTransformer] = None
nli_model: Optional[AutoModelForSequenceClassification] = None
nli_tokenizer: Optional[AutoTokenizer] = None
NLI_ENTAILMENT_LABEL_ID = -1 # Will be set after model loading

def load_nlp_models():
    global sentence_model, nli_model, nli_tokenizer, NLI_ENTAILMENT_LABEL_ID
    if sentence_model is None:
        logger.info("Loading Sentence Transformer model (all-mpnet-base-v2)...")
        sentence_model = SentenceTransformer('all-mpnet-base-v2', device=str(device)) # device can be 'cpu', 'cuda'
        logger.info("Sentence Transformer model loaded.")
    if nli_model is None or nli_tokenizer is None:
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        logger.info(f"Loading NLI model ({model_name})...")
        nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        nli_model.eval() # Set to evaluation mode
        logger.info("NLI model and tokenizer loaded.")

        # Determine the ID for the 'entailment' label
        # The order is typically [contradiction, neutral, entailment] for many MNLI models, but always check config
        # For MoritzLaurer models, they often use ["entailment", "neutral", "contradiction"] (0, 1, 2)
        # It's safer to check config.label2id
        if 'entailment' in nli_model.config.label2id:
            NLI_ENTAILMENT_LABEL_ID = nli_model.config.label2id['entailment']
        else: # Fallback for models that might have different label orders (less common for these specific models)
            logger.warning("Could not automatically determine 'entailment' label ID. Assuming order: c, n, e.")
            # Default for many RoBERTa/DeBERTa MNLI: 0: contradiction, 1: neutral, 2: entailment
            # MoritzLaurer models usually have entailment as 0. Check specific model card if unsure.
            # For "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", it's usually entailment:0, neutral:1, contradiction:2
            # Let's verify this. The template in the problem description implies ["entailment", "neutral", "contradiction"]
            # The model's config.json on Hugging Face hub confirms: "id2label": { "0": "entailment", "1": "neutral", "2": "contradiction" }
            NLI_ENTAILMENT_LABEL_ID = 0 # Based on MoritzLaurer's typical configuration and problem description
        logger.info(f"NLI entailment label ID set to: {NLI_ENTAILMENT_LABEL_ID}")


# Load models at application startup (for the main FastAPI process)
# For Celery workers, models should be loaded when the worker starts.
@app.on_event("startup")
async def startup_event():
    load_nlp_models()
    with SessionLocal() as db_startup: # Renamed to avoid conflict
        initialize_esg_topics(db_startup)

# Signal for Celery worker to load models
@worker_ready.connect
def on_worker_ready(**kwargs):
    logger.info("Celery worker ready. Loading NLP models...")
    load_nlp_models()
    logger.info("NLP models loaded in Celery worker.")


# --- Password Hashing & JWT ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Database Models (SQLAlchemy) ---
# --- Database Models (SQLAlchemy) ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False) # CORRECTED: Use sqlalchemy.String
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    reports = relationship("Report", back_populates="owner")

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filepath = Column(String)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded") # uploaded, nlp_processing, processed, reviewing, completed
    company_name = Column(String, nullable=True)
    final_score = Column(Integer, nullable=True, default=0)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="reports")
    annotations = relationship("ReportTopicAnnotation", back_populates="report", cascade="all, delete-orphan")
    chunks = relationship("ReportChunk", back_populates="report", cascade="all, delete-orphan")

class ESGTopic(Base):
    __tablename__ = "esg_topics"
    id = Column(Integer, primary_key=True, index=True)
    topic_number = Column(Integer, unique=True)
    name = Column(String, unique=True)
    description = Column(Text, nullable=True)
    hypothesis_template = Column(Text)

class ReportChunk(Base):
    __tablename__ = "report_chunks"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    chunk_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    # Store embedding as BLOB. Use numpy for efficient serialization.
    embedding = Column(LargeBinary, nullable=True) # Store serialized numpy array
    report = relationship("Report", back_populates="chunks")

    def set_embedding(self, vector: List[float]):
        # Serialize list of floats to bytes using numpy
        self.embedding = np.array(vector, dtype=np.float32).tobytes()

    def get_embedding(self) -> Optional[List[float]]:
        if self.embedding:
            # Deserialize bytes to numpy array, then to list
            return np.frombuffer(self.embedding, dtype=np.float32).tolist()
        return None

class ReportTopicAnnotation(Base):
    __tablename__ = "report_topic_annotations"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    topic_id = Column(Integer, ForeignKey("esg_topics.id"))
    status = Column(String, default="pending")  # "answered", "unanswered", "pending"
    auditor_notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    report = relationship("Report", back_populates="annotations")
    topic = relationship("ESGTopic")

Base.metadata.create_all(bind=engine)

# --- Pydantic Schemas ---
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_active: bool
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ESGTopicResponse(BaseModel):
    id: int
    topic_number: int
    name: str
    description: Optional[str] = None
    hypothesis_template: str
    class Config:
        from_attributes = True

class ReportTopicAnnotationCreate(BaseModel):
    status: str
    auditor_notes: Optional[str] = None

class ReportTopicAnnotationResponse(BaseModel):
    id: int
    topic_id: int
    status: str
    auditor_notes: Optional[str] = None
    timestamp: datetime
    topic: ESGTopicResponse
    class Config:
        from_attributes = True

class ReportResponse(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    status: str
    company_name: Optional[str] = None
    final_score: Optional[int] = 0
    user_id: int
    annotations: List[ReportTopicAnnotationResponse] = []
    class Config:
        from_attributes = True

class Suggestion(BaseModel):
    chunk_id: int
    chunk_text: str
    page_number: int
    entailment_score: float

class ReportChunkResponse(BaseModel): # For potential debugging
    id: int
    report_id: int
    page_number: int
    chunk_text_preview: str # To avoid sending huge texts
    has_embedding: bool
    class Config:
        from_attributes = True

# --- Dependency Injection & Auth ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Predefined ESG Topics ---
PREDEFINED_ESG_TOPICS = [
    {"topic_number": 1, "name": "GHG Emissions Scope 1", "description": "Disclosure of Scope 1 Greenhouse Gas emissions.", "hypothesis_template": "The company discloses its Scope 1 Greenhouse Gas emissions."},
    {"topic_number": 2, "name": "GHG Emissions Scope 2", "description": "Disclosure of Scope 2 Greenhouse Gas emissions.", "hypothesis_template": "The company discloses its Scope 2 Greenhouse Gas emissions."},
    {"topic_number": 3, "name": "Water Consumption", "description": "Disclosure of total water consumption.", "hypothesis_template": "The company discloses its total water consumption metrics."},
    # ... Add all 69 topics here ...
    {"topic_number": 69, "name": "Data Privacy and Security Policy", "description": "Disclosure of policies related to customer data privacy and security.", "hypothesis_template": "The company has a policy on customer data privacy and security."},
]

def initialize_esg_topics(db: Session):
    if db.query(ESGTopic).count() == 0:
        logger.info("Initializing ESG topics...")
        for topic_data in PREDEFINED_ESG_TOPICS:
            db_topic = ESGTopic(**topic_data)
            db.add(db_topic)
        db.commit()
        logger.info(f"{len(PREDEFINED_ESG_TOPICS)} ESG topics initialized.")

# --- Celery Tasks ---
@celery_app.task(name="tasks.process_report_nlp")
def process_report_nlp(report_id: int, filepath: str):
    logger.info(f"Celery task started: Processing NLP for report ID {report_id} from {filepath}")
    db: Session = SessionLocal()
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        logger.error(f"Report ID {report_id} not found in database for NLP processing.")
        db.close()
        return

    try:
        report.status = "nlp_processing"
        db.commit()

        # Ensure models are loaded in the worker (if not already by worker_ready)
        if sentence_model is None or nli_model is None: # Should be loaded by worker_ready
            logger.warning("NLP models not loaded in worker, attempting to load now.")
            load_nlp_models() # This might be problematic if worker_ready didn't fire or models are very large

        # 1. PDF Text Extraction and Chunking
        doc = fitz.open(filepath)
        chunks_created_count = 0
        for page_num_idx, page in enumerate(doc):
            page_num = page_num_idx + 1 # 1-based indexing for pages
            # Using `get_text("blocks")` can give more structured data including coordinates
            # For simplicity, we'll use paragraphs as chunks. A more robust method might be needed.
            text_blocks = page.get_text("blocks") # list of (x0, y0, x1, y1, "lines in block", block_no, block_type)
            for block in text_blocks:
                block_text = block[4].strip().replace("\n", " ") # Text content
                if len(block_text) > 50: # Basic filter for meaningful chunks
                    # 2. Generating embeddings for chunks
                    if sentence_model:
                        chunk_embedding_list = sentence_model.encode(block_text).tolist()
                    else:
                        logger.error("Sentence model not available in Celery worker!")
                        chunk_embedding_list = [] # Or handle error more gracefully

                    db_chunk = ReportChunk(
                        report_id=report_id,
                        chunk_text=block_text,
                        page_number=page_num
                    )
                    if chunk_embedding_list:
                         db_chunk.set_embedding(chunk_embedding_list) # Serialize and store
                    db.add(db_chunk)
                    chunks_created_count += 1
        doc.close()
        logger.info(f"Created {chunks_created_count} chunks for report ID {report_id}.")

        report.status = "processed" # NLP processing done
        db.commit()
        logger.info(f"NLP processing finished for report ID {report_id}. Status set to 'processed'.")

    except Exception as e:
        logger.error(f"Error during NLP processing for report ID {report_id}: {e}", exc_info=True)
        report.status = "nlp_failed"
        db.commit()
    finally:
        db.close()


# --- API Endpoints ---

# User Management & Auth
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
def create_user_endpoint(user_data: UserCreate, db: Session = Depends(get_db)): # Renamed user to user_data
    db_user = db.query(User).filter(User.email == user_data.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user_by_username = db.query(User).filter(User.username == user_data.username).first()
    if db_user_by_username:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = get_password_hash(user_data.password)
    db_user = User(username=user_data.username, email=user_data.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token, tags=["Users"])
async def login_for_access_token_endpoint(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse, tags=["Users"])
async def read_users_me_endpoint(current_user: User = Depends(get_current_active_user)):
    return current_user

# ESG Topics
@app.get("/esg_topics/", response_model=List[ESGTopicResponse], tags=["ESG Topics"])
def get_esg_topics_endpoint(db: Session = Depends(get_db)): # Renamed to avoid conflict
    topics = db.query(ESGTopic).order_by(ESGTopic.topic_number).all()
    return topics

# Report Management
@app.post("/reports/", response_model=ReportResponse, status_code=status.HTTP_201_CREATED, tags=["Reports"])
async def upload_report_endpoint( # Renamed
    company_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    # Sanitize filename (basic)
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_', '-')).strip()
    if not safe_filename:
        safe_filename = f"report_{datetime.utcnow().timestamp()}.pdf"
    
    # Ensure unique filename for storage to prevent overwrites
    timestamp_prefix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    unique_filename = f"{timestamp_prefix}_{safe_filename}"
    file_location = os.path.join(UPLOAD_DIRECTORY, unique_filename)

    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        logger.error(f"Could not save file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        if file.file:
             file.file.close()

    db_report = Report(
        filename=file.filename, # Original filename for display
        original_filepath=file_location, # Path to the stored unique file
        company_name=company_name,
        user_id=current_user.id,
        status="uploaded" # Initial status
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    # Initialize pending annotations for all ESG topics for this report
    all_topics = db.query(ESGTopic).all()
    for topic in all_topics:
        annotation = ReportTopicAnnotation(report_id=db_report.id, topic_id=topic.id, status="pending")
        db.add(annotation)
    db.commit() # Commit annotations

    # Trigger asynchronous NLP processing
    logger.info(f"Report {db_report.filename} (ID: {db_report.id}) uploaded by user {current_user.username}. Triggering NLP processing task.")
    process_report_nlp.delay(db_report.id, file_location)
    db_report.status = "nlp_queued" # Update status to show it's in queue
    db.commit()
    db.refresh(db_report) # Get the latest status

    # Fetch the report with its initialized annotations for the response
    report_with_annotations = db.query(Report)\
        .options(selectinload(Report.annotations).selectinload(ReportTopicAnnotation.topic))\
        .filter(Report.id == db_report.id).one()
    return report_with_annotations


@app.get("/reports/", response_model=List[ReportResponse], tags=["Reports"])
def get_reports_endpoint(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    reports = db.query(Report).filter(Report.user_id == current_user.id)\
                .options(selectinload(Report.annotations).selectinload(ReportTopicAnnotation.topic))\
                .order_by(Report.upload_timestamp.desc()).all()
    return reports

@app.get("/reports/{report_id}/", response_model=ReportResponse, tags=["Reports"])
def get_report_details_endpoint(report_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    report = db.query(Report).filter(Report.id == report_id, Report.user_id == current_user.id)\
               .options(selectinload(Report.annotations).selectinload(ReportTopicAnnotation.topic))\
               .first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@app.post("/reports/{report_id}/topics/{topic_id}/annotate/", response_model=ReportTopicAnnotationResponse, tags=["Annotations"])
def annotate_topic_endpoint( # Renamed
    report_id: int,
    topic_id: int,
    annotation_data: ReportTopicAnnotationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    report = db.query(Report).filter(Report.id == report_id, Report.user_id == current_user.id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    topic = db.query(ESGTopic).filter(ESGTopic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="ESG Topic not found")

    if annotation_data.status not in ["answered", "unanswered", "pending"]:
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'answered', 'unanswered', or 'pending'.")

    annotation = db.query(ReportTopicAnnotation).filter(
        ReportTopicAnnotation.report_id == report_id,
        ReportTopicAnnotation.topic_id == topic_id
    ).first()

    if annotation:
        annotation.status = annotation_data.status
        annotation.auditor_notes = annotation_data.auditor_notes
        # timestamp is onupdate
    else: # Should be pre-initialized, but as a fallback
        annotation = ReportTopicAnnotation(
            report_id=report_id, topic_id=topic_id,
            status=annotation_data.status, auditor_notes=annotation_data.auditor_notes
        )
        db.add(annotation)
    db.commit()

    # Update report score
    answered_count = db.query(ReportTopicAnnotation).filter(
        ReportTopicAnnotation.report_id == report_id,
        ReportTopicAnnotation.status == "answered"
    ).count()
    report.final_score = answered_count
    db.commit()
    db.refresh(annotation)
    # Eager load topic for response
    updated_annotation = db.query(ReportTopicAnnotation)\
        .options(selectinload(ReportTopicAnnotation.topic))\
        .filter(ReportTopicAnnotation.id == annotation.id).one()
    return updated_annotation


@app.get("/reports/{report_id}/score/", response_model=Dict[str, Any], tags=["Scores"])
def get_report_score_endpoint(report_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    report = db.query(Report).filter(Report.id == report_id, Report.user_id == current_user.id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    answered_count = db.query(ReportTopicAnnotation).filter(
        ReportTopicAnnotation.report_id == report_id,
        ReportTopicAnnotation.status == "answered"
    ).count()

    if report.final_score != answered_count:
        report.final_score = answered_count
        db.commit()
    max_score = db.query(ESGTopic).count()
    return {"report_id": report_id, "score": answered_count, "max_score": max_score, "status": report.status}

# NLP Suggestions Endpoint
@app.get("/reports/{report_id}/topics/{topic_id}/suggestions/", response_model=List[Suggestion], tags=["NLP Suggestions"])
async def get_nlp_suggestions_endpoint( # Renamed
    report_id: int,
    topic_id: int,
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    top_k_semantic: int = Query(20, ge=1, le=100), # Top K for semantic search
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    global sentence_model, nli_model, nli_tokenizer, NLI_ENTAILMENT_LABEL_ID # Ensure global models are used

    report = db.query(Report).filter(Report.id == report_id, Report.user_id == current_user.id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.status not in ["processed", "reviewing", "completed"]: # Check if NLP processing is done
         raise HTTPException(status_code=400, detail=f"Report NLP status is '{report.status}'. Suggestions available after 'processed'.")

    topic = db.query(ESGTopic).filter(ESGTopic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="ESG Topic not found")

    # Ensure models are loaded (primarily for cases where app startup might not have completed fully, or if called directly)
    if sentence_model is None or nli_model is None or nli_tokenizer is None or NLI_ENTAILMENT_LABEL_ID == -1:
        logger.warning("NLP models not loaded, attempting to load now for suggestions endpoint.")
        load_nlp_models() # This is a safeguard
        if sentence_model is None or nli_model is None or nli_tokenizer is None or NLI_ENTAILMENT_LABEL_ID == -1:
             raise HTTPException(status_code=503, detail="NLP models are not available. Please try again later.")


    hypothesis = topic.hypothesis_template
    logger.info(f"Generating suggestions for Report ID {report_id}, Topic ID {topic_id} ('{topic.name}') with threshold {threshold}, hypothesis: '{hypothesis}'")

    # 1. Retrieve report chunks and their embeddings
    report_chunks = db.query(ReportChunk).filter(ReportChunk.report_id == report_id).all()
    if not report_chunks:
        return []

    chunk_texts = [chunk.chunk_text for chunk in report_chunks]
    chunk_embeddings_list = [chunk.get_embedding() for chunk in report_chunks]
    
    valid_chunks_indices = [i for i, emb in enumerate(chunk_embeddings_list) if emb is not None]
    if not valid_chunks_indices:
        logger.warning(f"No valid embeddings found for chunks in report {report_id}")
        return []

    # Filter to use only chunks with valid embeddings
    filtered_chunk_texts = [chunk_texts[i] for i in valid_chunks_indices]
    filtered_report_chunks = [report_chunks[i] for i in valid_chunks_indices]
    # Convert list of lists to numpy array for sentence_transformers.util.semantic_search
    filtered_chunk_embeddings_np = np.array([chunk_embeddings_list[i] for i in valid_chunks_indices], dtype=np.float32)


    # 2. Stage 1: Candidate Retrieval (Sentence Transformer)
    hypothesis_embedding = sentence_model.encode(hypothesis, convert_to_tensor=True).cpu().numpy() # Ensure it's a numpy array

    # Using sentence_transformers.util.semantic_search for efficiency
    from sentence_transformers.util import semantic_search
    # semantic_search expects corpus_embeddings to be a 2D Tensor or ndarray
    hits = semantic_search(hypothesis_embedding.reshape(1, -1), filtered_chunk_embeddings_np, top_k=top_k_semantic)[0] # hits for the first query

    candidate_chunks_info = []
    for hit in hits:
        corpus_id = hit['corpus_id']
        original_chunk = filtered_report_chunks[corpus_id]
        candidate_chunks_info.append({
            "id": original_chunk.id,
            "text": original_chunk.chunk_text,
            "page_number": original_chunk.page_number,
            "semantic_score": hit['score']
        })
    
    if not candidate_chunks_info:
        logger.info("No semantic candidates found.")
        return []

    logger.info(f"Found {len(candidate_chunks_info)} semantic candidates.")

    # 3. Stage 2: NLI Inference (DeBERTa NLI)
    suggestions = []
    nli_premises = [candidate['text'] for candidate in candidate_chunks_info]
    nli_hypotheses = [hypothesis] * len(nli_premises) # Repeat hypothesis for each premise

    # Batch NLI processing
    if nli_premises:
        try:
            inputs = nli_tokenizer(nli_premises, nli_hypotheses, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            with torch.no_grad():
                outputs = nli_model(**inputs)
            
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

            for idx, candidate_info in enumerate(candidate_chunks_info):
                entailment_prob = probabilities[idx][NLI_ENTAILMENT_LABEL_ID] # Use determined entailment ID
                if entailment_prob >= threshold:
                    suggestions.append(Suggestion(
                        chunk_id=candidate_info["id"],
                        chunk_text=candidate_info["text"][:500] + "..." if len(candidate_info["text"]) > 500 else candidate_info["text"], # Preview
                        page_number=candidate_info["page_number"],
                        entailment_score=round(entailment_prob, 4)
                    ))
        except Exception as e:
            logger.error(f"Error during NLI batch inference: {e}", exc_info=True)
            # Optionally try one by one or just fail
            raise HTTPException(status_code=500, detail="Error during NLI inference.")


    logger.info(f"Generated {len(suggestions)} suggestions passing threshold {threshold}.")
    # Sort suggestions by entailment score (descending)
    suggestions.sort(key=lambda s: s.entailment_score, reverse=True)
    return suggestions

# --- Main (for Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ESG Scoring Assistant API...")
    # Base.metadata.drop_all(bind=engine) # Uncomment to clear DB on restart during dev
    # Base.metadata.create_all(bind=engine) # Ensure tables are created
    # with SessionLocal() as db_main:
    #     initialize_esg_topics(db_main)
    uvicorn.run(app, host="0.0.0.0", port=8000)