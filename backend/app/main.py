import os
import shutil
import logging
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timedelta
import json  # For serializing list of floats for embeddings
import numpy as np  # For efficient serialization of embeddings
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    status,
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse  # Added for serving PDF files
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    Boolean,
    Text,
    DateTime,
    LargeBinary,
)
from sqlalchemy.orm import (
    sessionmaker,
    Session,
    relationship,
    declarative_base,
    selectinload,
)
from pydantic import BaseModel, Field, EmailStr

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
from celery.signals import worker_ready  # To load models in worker

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

celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
NLI_ENTAILMENT_LABEL_ID = -1  # Will be set after model loading


def load_nlp_models():
    global sentence_model, nli_model, nli_tokenizer, NLI_ENTAILMENT_LABEL_ID
    if sentence_model is None:
        logger.info("Loading Sentence Transformer model (all-mpnet-base-v2)...")
        sentence_model = SentenceTransformer(
            "all-mpnet-base-v2", device=str(device)
        )  # device can be 'cpu', 'cuda'
        logger.info("Sentence Transformer model loaded.")
    if nli_model is None or nli_tokenizer is None:
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        logger.info(f"Loading NLI model ({model_name})...")
        nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        nli_model.eval()  # Set to evaluation mode
        logger.info("NLI model and tokenizer loaded.")

        if "entailment" in nli_model.config.label2id:
            NLI_ENTAILMENT_LABEL_ID = nli_model.config.label2id["entailment"]
        else:
            logger.warning(
                "Could not automatically determine 'entailment' label ID from model config. Assuming 'entailment' is label 0 for MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli."
            )
            NLI_ENTAILMENT_LABEL_ID = (
                0  # Based on MoritzLaurer's typical configuration for this model
            )
        logger.info(f"NLI entailment label ID set to: {NLI_ENTAILMENT_LABEL_ID}")


@app.on_event("startup")
async def startup_event():
    load_nlp_models()
    Base.metadata.create_all(bind=engine)  # Ensure tables are created on startup
    with SessionLocal() as db_startup:
        initialize_esg_topics(db_startup)


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
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    reports = relationship("Report", back_populates="owner")

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filepath = Column(String)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded")
    nlp_progress = Column(Integer, default=0) # << NEW FIELD (0-100)
    company_name = Column(String, nullable=True)
    final_score = Column(Integer, nullable=True, default=0)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="reports")
    annotations = relationship(
        "ReportTopicAnnotation", back_populates="report", cascade="all, delete-orphan"
    )
    chunks = relationship(
        "ReportChunk", back_populates="report", cascade="all, delete-orphan"
    )

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
    coordinates_json = Column(
        String, nullable=True
    )  # For PDF highlighting: Store as JSON: {"x0": ..., "y0": ..., "x1": ..., "y1": ...}
    embedding = Column(LargeBinary, nullable=True)
    report = relationship("Report", back_populates="chunks")

    def set_embedding(self, vector: List[float]):
        self.embedding = np.array(vector, dtype=np.float32).tobytes()

    def get_embedding(self) -> Optional[List[float]]:
        if self.embedding:
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


# Base.metadata.create_all(bind=engine) # Moved to startup_event


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
    status: Literal["answered", "unanswered", "pending"]
    auditor_notes: Optional[str] = None


class ReportTopicAnnotationResponse(BaseModel):
    id: int
    topic_id: int
    status: str
    auditor_notes: Optional[str] = None
    timestamp: datetime
    topic: ESGTopicResponse  # Eager load topic info

    class Config:
        from_attributes = True

class ReportResponse(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    status: str
    nlp_progress: Optional[int] = 0 # << NEW FIELD
    company_name: Optional[str] = None
    final_score: Optional[int] = 0
    user_id: int
    annotations: List[ReportTopicAnnotationResponse] = []

    class Config:
        from_attributes = True

class ReportStatusUpdate(BaseModel):  # New schema for updating report status
    status: Literal[
        "uploaded",
        "nlp_queued",
        "nlp_processing",
        "processed",
        "reviewing",
        "completed",
        "nlp_failed",
    ]


class Suggestion(BaseModel):
    chunk_id: int
    chunk_text: str
    page_number: int
    entailment_score: float
    coordinates: Optional[Dict[str, float]] = None  # For PDF highlighting


class ReportChunkResponse(BaseModel):
    id: int
    report_id: int
    page_number: int
    chunk_text: str
    coordinates: Optional[Dict[str, float]] = None
    has_embedding: bool

    class Config:
        from_attributes = True


class HealthCheck(BaseModel):
    status: str
    models_loaded: bool
    database_connected: bool


# --- Dependency Injection & Auth ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
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


# --- Complete ESG Topics (All 69 topics) ---
PREDEFINED_ESG_TOPICS = [
    {
        "topic_number": 1,
        "name": "GHG Emissions Scope 1",
        "description": "Disclosure of Scope 1 Greenhouse Gas emissions.",
        "hypothesis_template": "The company discloses its Scope 1 Greenhouse Gas emissions.",
    },
    {
        "topic_number": 2,
        "name": "GHG Emissions Scope 2",
        "description": "Disclosure of Scope 2 Greenhouse Gas emissions.",
        "hypothesis_template": "The company discloses its Scope 2 Greenhouse Gas emissions.",
    },
    {
        "topic_number": 3,
        "name": "Water Consumption",
        "description": "Disclosure of total water consumption.",
        "hypothesis_template": "The company discloses its total water consumption metrics.",
    },
    {
        "topic_number": 4,
        "name": "Energy Consumption",
        "description": "Disclosure of total energy consumption.",
        "hypothesis_template": "The company discloses its total energy consumption.",
    },
    {
        "topic_number": 5,
        "name": "Renewable Energy Usage",
        "description": "Disclosure of renewable energy usage.",
        "hypothesis_template": "The company reports on its use of renewable energy sources.",
    },
    {
        "topic_number": 6,
        "name": "Waste Generation",
        "description": "Disclosure of waste generation data.",
        "hypothesis_template": "The company discloses information about waste generation.",
    },
    {
        "topic_number": 7,
        "name": "Waste Recycling",
        "description": "Disclosure of waste recycling practices.",
        "hypothesis_template": "The company reports on its waste recycling initiatives.",
    },
    {
        "topic_number": 8,
        "name": "Biodiversity Conservation",
        "description": "Disclosure of biodiversity conservation efforts.",
        "hypothesis_template": "The company describes its biodiversity conservation initiatives.",
    },
    {
        "topic_number": 9,
        "name": "Environmental Compliance",
        "description": "Disclosure of environmental compliance status.",
        "hypothesis_template": "The company reports on its environmental compliance status.",
    },
    {
        "topic_number": 10,
        "name": "Environmental Management System",
        "description": "Disclosure of environmental management systems.",
        "hypothesis_template": "The company has an environmental management system in place.",
    },
    {
        "topic_number": 11,
        "name": "Carbon Footprint Reduction",
        "description": "Disclosure of carbon footprint reduction initiatives.",
        "hypothesis_template": "The company reports on carbon footprint reduction efforts.",
    },
    {
        "topic_number": 12,
        "name": "Sustainable Procurement",
        "description": "Disclosure of sustainable procurement practices.",
        "hypothesis_template": "The company has sustainable procurement policies.",
    },
    {
        "topic_number": 13,
        "name": "Green Building Practices",
        "description": "Disclosure of green building initiatives.",
        "hypothesis_template": "The company implements green building practices.",
    },
    {
        "topic_number": 14,
        "name": "Air Quality Management",
        "description": "Disclosure of air quality management measures.",
        "hypothesis_template": "The company manages and reports on air quality.",
    },
    {
        "topic_number": 15,
        "name": "Soil Conservation",
        "description": "Disclosure of soil conservation measures.",
        "hypothesis_template": "The company implements soil conservation measures.",
    },
    {
        "topic_number": 16,
        "name": "Employee Health and Safety",
        "description": "Disclosure of employee health and safety measures.",
        "hypothesis_template": "The company reports on employee health and safety programs.",
    },
    {
        "topic_number": 17,
        "name": "Training and Development",
        "description": "Disclosure of employee training and development programs.",
        "hypothesis_template": "The company provides training and development programs for employees.",
    },
    {
        "topic_number": 18,
        "name": "Diversity and Inclusion",
        "description": "Disclosure of diversity and inclusion initiatives.",
        "hypothesis_template": "The company has diversity and inclusion policies and programs.",
    },
    {
        "topic_number": 19,
        "name": "Employee Benefits",
        "description": "Disclosure of employee benefits and welfare programs.",
        "hypothesis_template": "The company provides comprehensive employee benefits.",
    },
    {
        "topic_number": 20,
        "name": "Fair Labor Practices",
        "description": "Disclosure of fair labor practices.",
        "hypothesis_template": "The company ensures fair labor practices across its operations.",
    },
    {
        "topic_number": 21,
        "name": "Community Engagement",
        "description": "Disclosure of community engagement activities.",
        "hypothesis_template": "The company actively engages with local communities.",
    },
    {
        "topic_number": 22,
        "name": "Stakeholder Engagement",
        "description": "Disclosure of stakeholder engagement processes.",
        "hypothesis_template": "The company has formal stakeholder engagement processes.",
    },
    {
        "topic_number": 23,
        "name": "Customer Satisfaction",
        "description": "Disclosure of customer satisfaction measures.",
        "hypothesis_template": "The company measures and reports on customer satisfaction.",
    },
    {
        "topic_number": 24,
        "name": "Product Quality and Safety",
        "description": "Disclosure of product quality and safety measures.",
        "hypothesis_template": "The company ensures product quality and safety.",
    },
    {
        "topic_number": 25,
        "name": "Supply Chain Management",
        "description": "Disclosure of supply chain management practices.",
        "hypothesis_template": "The company has responsible supply chain management practices.",
    },
    {
        "topic_number": 26,
        "name": "Human Rights Policy",
        "description": "Disclosure of human rights policies.",
        "hypothesis_template": "The company has a human rights policy.",
    },
    {
        "topic_number": 27,
        "name": "Anti-Corruption Measures",
        "description": "Disclosure of anti-corruption and anti-bribery measures.",
        "hypothesis_template": "The company has anti-corruption and anti-bribery policies.",
    },
    {
        "topic_number": 28,
        "name": "Whistleblower Protection",
        "description": "Disclosure of whistleblower protection mechanisms.",
        "hypothesis_template": "The company has whistleblower protection mechanisms.",
    },
    {
        "topic_number": 29,
        "name": "Board Independence",
        "description": "Disclosure of board independence measures.",
        "hypothesis_template": "The company maintains board independence.",
    },
    {
        "topic_number": 30,
        "name": "Board Diversity",
        "description": "Disclosure of board diversity initiatives.",
        "hypothesis_template": "The company promotes diversity in its board composition.",
    },
    {
        "topic_number": 31,
        "name": "Executive Compensation",
        "description": "Disclosure of executive compensation policies.",
        "hypothesis_template": "The company discloses executive compensation policies.",
    },
    {
        "topic_number": 32,
        "name": "Audit Committee Independence",
        "description": "Disclosure of audit committee independence.",
        "hypothesis_template": "The company maintains audit committee independence.",
    },
    {
        "topic_number": 33,
        "name": "Risk Management Framework",
        "description": "Disclosure of risk management frameworks.",
        "hypothesis_template": "The company has a comprehensive risk management framework.",
    },
    {
        "topic_number": 34,
        "name": "Internal Controls",
        "description": "Disclosure of internal control systems.",
        "hypothesis_template": "The company has effective internal control systems.",
    },
    {
        "topic_number": 35,
        "name": "Regulatory Compliance",
        "description": "Disclosure of regulatory compliance status.",
        "hypothesis_template": "The company maintains regulatory compliance.",
    },
    {
        "topic_number": 36,
        "name": "Shareholder Rights",
        "description": "Disclosure of shareholder rights protection.",
        "hypothesis_template": "The company protects shareholder rights.",
    },
    {
        "topic_number": 37,
        "name": "Transparency and Disclosure",
        "description": "Disclosure of transparency and reporting practices.",
        "hypothesis_template": "The company maintains transparency in its operations and reporting.",
    },
    {
        "topic_number": 38,
        "name": "Cybersecurity Measures",
        "description": "Disclosure of cybersecurity policies and measures.",
        "hypothesis_template": "The company has cybersecurity policies and measures in place.",
    },
    {
        "topic_number": 39,
        "name": "Innovation and R&D",
        "description": "Disclosure of innovation and research & development activities.",
        "hypothesis_template": "The company invests in innovation and research & development.",
    },
    {
        "topic_number": 40,
        "name": "Digital Transformation",
        "description": "Disclosure of digital transformation initiatives.",
        "hypothesis_template": "The company is undertaking digital transformation initiatives.",
    },
    {
        "topic_number": 41,
        "name": "Intellectual Property Protection",
        "description": "Disclosure of intellectual property protection measures.",
        "hypothesis_template": "The company protects its intellectual property.",
    },
    {
        "topic_number": 42,
        "name": "Crisis Management",
        "description": "Disclosure of crisis management preparedness.",
        "hypothesis_template": "The company has crisis management protocols in place.",
    },
    {
        "topic_number": 43,
        "name": "Business Continuity Planning",
        "description": "Disclosure of business continuity planning.",
        "hypothesis_template": "The company has business continuity plans.",
    },
    {
        "topic_number": 44,
        "name": "Succession Planning",
        "description": "Disclosure of leadership succession planning.",
        "hypothesis_template": "The company has succession planning for key leadership positions.",
    },
    {
        "topic_number": 45,
        "name": "Performance Measurement",
        "description": "Disclosure of performance measurement systems.",
        "hypothesis_template": "The company has performance measurement systems in place.",
    },
    {
        "topic_number": 46,
        "name": "Financial Reporting Quality",
        "description": "Disclosure of financial reporting quality measures.",
        "hypothesis_template": "The company ensures high quality financial reporting.",
    },
    {
        "topic_number": 47,
        "name": "Tax Strategy",
        "description": "Disclosure of tax strategy and compliance.",
        "hypothesis_template": "The company has a clear tax strategy and compliance framework.",
    },
    {
        "topic_number": 48,
        "name": "Capital Allocation",
        "description": "Disclosure of capital allocation policies.",
        "hypothesis_template": "The company has clear capital allocation policies.",
    },
    {
        "topic_number": 49,
        "name": "Merger and Acquisition Policy",
        "description": "Disclosure of merger and acquisition policies.",
        "hypothesis_template": "The company has policies for mergers and acquisitions.",
    },
    {
        "topic_number": 50,
        "name": "Partnership Strategy",
        "description": "Disclosure of strategic partnership approaches.",
        "hypothesis_template": "The company has a strategic approach to partnerships.",
    },
    {
        "topic_number": 51,
        "name": "Market Expansion Strategy",
        "description": "Disclosure of market expansion strategies.",
        "hypothesis_template": "The company has market expansion strategies.",
    },
    {
        "topic_number": 52,
        "name": "Competitive Positioning",
        "description": "Disclosure of competitive positioning strategy.",
        "hypothesis_template": "The company discloses its competitive positioning strategy.",
    },
    {
        "topic_number": 53,
        "name": "Brand Management",
        "description": "Disclosure of brand management practices.",
        "hypothesis_template": "The company has brand management practices in place.",
    },
    {
        "topic_number": 54,
        "name": "Customer Data Protection",
        "description": "Disclosure of customer data protection measures.",
        "hypothesis_template": "The company protects customer data and privacy.",
    },
    {
        "topic_number": 55,
        "name": "Supplier Diversity",
        "description": "Disclosure of supplier diversity programs.",
        "hypothesis_template": "The company has supplier diversity programs.",
    },
    {
        "topic_number": 56,
        "name": "Local Sourcing",
        "description": "Disclosure of local sourcing initiatives.",
        "hypothesis_template": "The company prioritizes local sourcing where possible.",
    },
    {
        "topic_number": 57,
        "name": "Economic Impact Assessment",
        "description": "Disclosure of economic impact on communities.",
        "hypothesis_template": "The company assesses its economic impact on communities.",
    },
    {
        "topic_number": 58,
        "name": "Employment Generation",
        "description": "Disclosure of employment generation initiatives.",
        "hypothesis_template": "The company contributes to employment generation.",
    },
    {
        "topic_number": 59,
        "name": "Skill Development Programs",
        "description": "Disclosure of skill development programs.",
        "hypothesis_template": "The company provides skill development programs.",
    },
    {
        "topic_number": 60,
        "name": "Educational Initiatives",
        "description": "Disclosure of educational support initiatives.",
        "hypothesis_template": "The company supports educational initiatives.",
    },
    {
        "topic_number": 61,
        "name": "Healthcare Initiatives",
        "description": "Disclosure of healthcare support programs.",
        "hypothesis_template": "The company supports healthcare initiatives.",
    },
    {
        "topic_number": 62,
        "name": "Rural Development",
        "description": "Disclosure of rural development programs.",
        "hypothesis_template": "The company contributes to rural development.",
    },
    {
        "topic_number": 63,
        "name": "Women Empowerment",
        "description": "Disclosure of women empowerment initiatives.",
        "hypothesis_template": "The company has women empowerment programs.",
    },
    {
        "topic_number": 64,
        "name": "Technology Transfer",
        "description": "Disclosure of technology transfer initiatives.",
        "hypothesis_template": "The company engages in technology transfer activities.",
    },
    {
        "topic_number": 65,
        "name": "Financial Inclusion",
        "description": "Disclosure of financial inclusion efforts.",
        "hypothesis_template": "The company promotes financial inclusion.",
    },
    {
        "topic_number": 66,
        "name": "Disaster Relief",
        "description": "Disclosure of disaster relief contributions.",
        "hypothesis_template": "The company contributes to disaster relief efforts.",
    },
    {
        "topic_number": 67,
        "name": "Sports and Culture Promotion",
        "description": "Disclosure of sports and cultural promotion activities.",
        "hypothesis_template": "The company promotes sports and cultural activities.",
    },
    {
        "topic_number": 68,
        "name": "Environmental Restoration",
        "description": "Disclosure of environmental restoration projects.",
        "hypothesis_template": "The company undertakes environmental restoration projects.",
    },
    {
        "topic_number": 69,
        "name": "Data Privacy and Security Policy",
        "description": "Disclosure of policies related to customer data privacy and security.",
        "hypothesis_template": "The company has a policy on customer data privacy and security.",
    },
]


def initialize_esg_topics(db: Session):
    if db.query(ESGTopic).count() < len(
        PREDEFINED_ESG_TOPICS
    ):  # Check if we need to add/update
        existing_topic_numbers = {
            t.topic_number for t in db.query(ESGTopic.topic_number).all()
        }
        logger.info("Initializing/Updating ESG topics...")
        topics_added_count = 0
        for topic_data in PREDEFINED_ESG_TOPICS:
            if topic_data["topic_number"] not in existing_topic_numbers:
                db_topic = ESGTopic(**topic_data)
                db.add(db_topic)
                topics_added_count += 1
            else:  # Optionally update existing topics if needed
                # db.query(ESGTopic).filter(ESGTopic.topic_number == topic_data["topic_number"]).update(topic_data)
                pass
        if topics_added_count > 0:
            db.commit()
            logger.info(f"{topics_added_count} ESG topics newly initialized/updated.")
        else:
            logger.info("ESG topics are already up to date.")


# --- Celery Tasks ---
@celery_app.task(name="tasks.process_report_nlp")
# In the process_report_nlp function, around line 879
def process_report_nlp(report_id: int, filepath: str):
    logger.info(
        f"Celery task started: Processing NLP for report ID {report_id} from {filepath}"
    )
    db: Session = SessionLocal()
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        logger.error(f"Report ID {report_id} not found in database for NLP processing.")
        db.close()
        return

    try:
        report.status = "nlp_processing"
        report.nlp_progress = 0
        db.commit()

        if sentence_model is None or nli_model is None:
            logger.warning("NLP models not loaded in worker (process_report_nlp), attempting to load now.")
            load_nlp_models()
            if sentence_model is None or nli_model is None:
                logger.error("Failed to load NLP models in Celery worker. Aborting NLP task.")
                report.status = "nlp_failed"; report.nlp_progress = 0
                db.commit(); db.close()
                return

        # Normalize the file path for the current OS
        normalized_filepath = os.path.normpath(filepath)
        logger.info(f"Normalized filepath: {normalized_filepath}")
        
        if not os.path.exists(normalized_filepath):
            logger.error(f"File does not exist: {normalized_filepath}")
            report.status = "nlp_failed"; report.nlp_progress = 0
            db.commit(); db.close()
            return

        doc = fitz.open(normalized_filepath)
        total_pages = len(doc)
        chunks_created_count = 0

        if total_pages == 0:
            logger.warning(f"Report ID {report_id} is an empty PDF (0 pages). Marking as processed.")
            report.status = "processed" # Or "nlp_failed" or a custom status
            report.nlp_progress = 100
            db.commit()
            doc.close()
            db.close()
            return

        for page_num_idx, page in enumerate(doc):
            page_num = page_num_idx + 1
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                block_text = block[4].strip().replace("\n", " ")
                if len(block_text) > 50: # Basic filter
                    chunk_embedding_list = sentence_model.encode(block_text).tolist() if sentence_model else []

                    db_chunk = ReportChunk(
                        report_id=report_id, chunk_text=block_text, page_number=page_num
                    )
                    block_coords = {"x0": block[0], "y0": block[1], "x1": block[2], "y1": block[3]}
                    db_chunk.coordinates_json = json.dumps(block_coords)
                    if chunk_embedding_list:
                        db_chunk.set_embedding(chunk_embedding_list)
                    db.add(db_chunk)
                    chunks_created_count += 1

            # Update progress after processing each page's blocks
            # We commit all chunks at the end of the loop for efficiency, but progress can be updated more often
            current_progress_percentage = int(((page_num_idx + 1) / total_pages) * 95) # Go up to 95%, 100% on final commit
            if report.nlp_progress < current_progress_percentage: # Update only if progress increased
                report.nlp_progress = current_progress_percentage
                try:
                    db.commit() # Commit progress update
                    db.refresh(report) # Refresh to get the latest state if needed elsewhere
                except Exception as progress_commit_exc:
                    logger.error(f"Error committing interim progress for report {report_id}: {progress_commit_exc}")
                    db.rollback() # Rollback only this progress commit attempt
                    # The main transaction for chunks will continue.

        db.commit() # Commit all added chunks
        doc.close()
        logger.info(f"Created {chunks_created_count} chunks for report ID {report_id}.")

        report.status = "processed"
        report.nlp_progress = 100 # Mark as 100%
        db.commit()
        logger.info(
            f"NLP processing finished for report ID {report_id}. Status set to 'processed'."
        )

    except Exception as e:
        logger.error(
            f"Error during NLP processing for report ID {report_id}: {e}", exc_info=True
        )
        if report: # Check if report was fetched
            report.status = "nlp_failed"
            report.nlp_progress = 0 # Or last known good progress, or a specific error progress value
            db.commit()
    finally:
        db.close()

# --- API Endpoints ---

# Health Check
@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    models_loaded = sentence_model is not None and nli_model is not None
    database_connected = True
    try:
        db.execute("SELECT 1")
    except Exception:
        database_connected = False

    return HealthCheck(
        status="healthy" if models_loaded and database_connected else "unhealthy",
        models_loaded=models_loaded,
        database_connected=database_connected,
    )


# User Management & Auth
@app.post(
    "/users/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Users"],
)
def create_user_endpoint(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user_data.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user_by_username = (
        db.query(User).filter(User.username == user_data.username).first()
    )
    if db_user_by_username:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = get_password_hash(user_data.password)
    db_user_instance = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
    )
    db.add(db_user_instance)
    db.commit()
    db.refresh(db_user_instance)
    return db_user_instance


@app.post("/token", response_model=Token, tags=["Users"])
async def login_for_access_token_endpoint(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
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
def get_esg_topics_endpoint(db: Session = Depends(get_db)):
    topics = db.query(ESGTopic).order_by(ESGTopic.topic_number).all()
    return topics


@app.get(
    "/esg_topics/{topic_id}/", response_model=ESGTopicResponse, tags=["ESG Topics"]
)
def get_esg_topic_endpoint(topic_id: int, db: Session = Depends(get_db)):
    topic = db.query(ESGTopic).filter(ESGTopic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="ESG Topic not found")
    return topic


# Report Management
@app.post(
    "/reports/",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Reports"],
)
async def upload_report_endpoint(
    company_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PDF files are allowed."
        )

    safe_filename = "".join(
        c for c in file.filename if c.isalnum() or c in (".", "_", "-")
    ).strip()
    if not safe_filename:
        safe_filename = f"report_{datetime.utcnow().timestamp()}.pdf"

    timestamp_prefix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    unique_filename = f"{timestamp_prefix}_{current_user.id}_{safe_filename}"  # Add user_id for better uniqueness
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
        filename=file.filename,
        original_filepath=file_location,
        company_name=company_name,
        user_id=current_user.id,
        status="uploaded",
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    all_topics = db.query(ESGTopic).all()
    for (
        topic_item
    ) in (
        all_topics
    ):  # Renamed topic to topic_item to avoid conflict with ReportTopicAnnotation.topic
        annotation = ReportTopicAnnotation(
            report_id=db_report.id, topic_id=topic_item.id, status="pending"
        )
        db.add(annotation)
    db.commit()

    logger.info(
        f"Report {db_report.filename} (ID: {db_report.id}) uploaded by user {current_user.username}. Triggering NLP processing task."
    )
    process_report_nlp.delay(db_report.id, file_location)
    db_report.status = "nlp_queued"
    db.commit()
    db.refresh(db_report)

    report_with_annotations = (
        db.query(Report)
        .options(
            selectinload(Report.annotations).selectinload(ReportTopicAnnotation.topic)
        )
        .filter(Report.id == db_report.id)
        .one()
    )
    return report_with_annotations


@app.get("/reports/", response_model=List[ReportResponse], tags=["Reports"])
def get_reports_endpoint(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)
):
    reports = (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .options(
            selectinload(Report.annotations).selectinload(ReportTopicAnnotation.topic)
        )
        .order_by(Report.upload_timestamp.desc())
        .all()
    )
    return reports


@app.get("/reports/{report_id}/", response_model=ReportResponse, tags=["Reports"])
def get_report_details_endpoint(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .options(
            selectinload(Report.annotations).selectinload(ReportTopicAnnotation.topic)
        )
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.delete(
    "/reports/{report_id}/", status_code=status.HTTP_204_NO_CONTENT, tags=["Reports"]
)
async def delete_report_endpoint(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Delete the physical file
    if report.original_filepath and os.path.exists(report.original_filepath):
        try:
            os.remove(report.original_filepath)
        except Exception as e:
            logger.warning(f"Could not delete file {report.original_filepath}: {e}")

    # Delete from database (cascade will handle chunks and annotations)
    db.delete(report)
    db.commit()


# Serve PDF file
@app.get("/reports/{report_id}/pdf", tags=["Reports"], response_class=FileResponse)
async def serve_report_pdf_endpoint(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found or access denied",
        )

    if not report.original_filepath or not os.path.exists(report.original_filepath):
        logger.error(
            f"PDF file not found for report ID {report_id} at path: {report.original_filepath}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="PDF file not found on server"
        )

    return FileResponse(
        path=report.original_filepath,
        media_type="application/pdf",
        filename=report.filename,
    )


# Update Report Status
VALID_REPORT_STATUSES_FOR_UPDATE = [
    "uploaded",
    "nlp_queued",
    "nlp_processing",
    "processed",
    "reviewing",
    "completed",
    "nlp_failed",
]


@app.patch(
    "/reports/{report_id}/status", response_model=ReportResponse, tags=["Reports"]
)
async def update_report_status_endpoint(
    report_id: int,
    status_update: ReportStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .options(
            selectinload(Report.annotations).selectinload(ReportTopicAnnotation.topic)
        )
        .first()
    )  # Eager load annotations for the response

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Report not found"
        )

    report.status = status_update.status
    db.commit()
    db.refresh(report)
    return report


# Chunk Management
@app.get(
    "/reports/{report_id}/chunks/",
    response_model=List[ReportChunkResponse],
    tags=["Chunks"],
)
async def get_report_chunks_endpoint(
    report_id: int,
    page: Optional[int] = Query(None, description="Filter by page number"),
    limit: int = Query(50, ge=1, le=500, description="Number of chunks to return"),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    query = db.query(ReportChunk).filter(ReportChunk.report_id == report_id)

    if page is not None:
        query = query.filter(ReportChunk.page_number == page)

    chunks = query.offset(offset).limit(limit).all()

    result = []
    for chunk in chunks:
        coordinates = None
        if chunk.coordinates_json:
            try:
                coordinates = json.loads(chunk.coordinates_json)
            except json.JSONDecodeError:
                pass

        result.append(
            ReportChunkResponse(
                id=chunk.id,
                report_id=chunk.report_id,
                page_number=chunk.page_number,
                chunk_text=chunk.chunk_text,
                coordinates=coordinates,
                has_embedding=chunk.embedding is not None,
            )
        )

    return result


@app.get(
    "/reports/{report_id}/chunks/{chunk_id}/",
    response_model=ReportChunkResponse,
    tags=["Chunks"],
)
async def get_chunk_details_endpoint(
    report_id: int,
    chunk_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Verify report ownership
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    chunk = (
        db.query(ReportChunk)
        .filter(ReportChunk.id == chunk_id, ReportChunk.report_id == report_id)
        .first()
    )

    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    coordinates = None
    if chunk.coordinates_json:
        try:
            coordinates = json.loads(chunk.coordinates_json)
        except json.JSONDecodeError:
            pass

    return ReportChunkResponse(
        id=chunk.id,
        report_id=chunk.report_id,
        page_number=chunk.page_number,
        chunk_text=chunk.chunk_text,
        coordinates=coordinates,
        has_embedding=chunk.embedding is not None,
    )


# Annotations
@app.post(
    "/reports/{report_id}/topics/{topic_id}/annotate/",
    response_model=ReportTopicAnnotationResponse,
    tags=["Annotations"],
)
def annotate_topic_endpoint(
    report_id: int,
    topic_id: int,
    annotation_data: ReportTopicAnnotationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    topic_item = (
        db.query(ESGTopic).filter(ESGTopic.id == topic_id).first()
    )  # Renamed topic to topic_item
    if not topic_item:
        raise HTTPException(status_code=404, detail="ESG Topic not found")

    annotation = (
        db.query(ReportTopicAnnotation)
        .filter(
            ReportTopicAnnotation.report_id == report_id,
            ReportTopicAnnotation.topic_id == topic_id,
        )
        .first()
    )

    if annotation:
        annotation.status = annotation_data.status
        annotation.auditor_notes = annotation_data.auditor_notes
        annotation.timestamp = datetime.utcnow()  # Update timestamp
    else:
        # This case should ideally not happen if annotations are pre-initialized
        annotation = ReportTopicAnnotation(
            report_id=report_id,
            topic_id=topic_id,
            status=annotation_data.status,
            auditor_notes=annotation_data.auditor_notes,
        )
        db.add(annotation)
    db.commit()

    answered_count = (
        db.query(ReportTopicAnnotation)
        .filter(
            ReportTopicAnnotation.report_id == report_id,
            ReportTopicAnnotation.status == "answered",
        )
        .count()
    )
    report.final_score = answered_count
    db.commit()

    # Refresh annotation to get any DB-side changes (like timestamp) and eager load its topic
    updated_annotation = (
        db.query(ReportTopicAnnotation)
        .options(selectinload(ReportTopicAnnotation.topic))
        .filter(ReportTopicAnnotation.id == annotation.id)
        .one()
    )
    return updated_annotation


@app.get(
    "/reports/{report_id}/annotations/",
    response_model=List[ReportTopicAnnotationResponse],
    tags=["Annotations"],
)
def get_report_annotations_endpoint(
    report_id: int,
    status_filter: Optional[Literal["answered", "unanswered", "pending"]] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    query = (
        db.query(ReportTopicAnnotation)
        .options(selectinload(ReportTopicAnnotation.topic))
        .filter(ReportTopicAnnotation.report_id == report_id)
    )

    if status_filter:
        query = query.filter(ReportTopicAnnotation.status == status_filter)

    annotations = query.all()
    return annotations


# Scores
@app.get("/reports/{report_id}/score/", response_model=Dict[str, Any], tags=["Scores"])
def get_report_score_endpoint(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    answered_count = (
        db.query(ReportTopicAnnotation)
        .filter(
            ReportTopicAnnotation.report_id == report_id,
            ReportTopicAnnotation.status == "answered",
        )
        .count()
    )

    if report.final_score != answered_count:  # Ensure score is up-to-date
        report.final_score = answered_count
        db.commit()
        db.refresh(report)

    max_score = db.query(ESGTopic).count()

    # Additional statistics
    pending_count = (
        db.query(ReportTopicAnnotation)
        .filter(
            ReportTopicAnnotation.report_id == report_id,
            ReportTopicAnnotation.status == "pending",
        )
        .count()
    )

    unanswered_count = (
        db.query(ReportTopicAnnotation)
        .filter(
            ReportTopicAnnotation.report_id == report_id,
            ReportTopicAnnotation.status == "unanswered",
        )
        .count()
    )

    return {
        "report_id": report_id,
        "score": report.final_score,
        "max_score": max_score,
        "percentage": (
            round((report.final_score / max_score) * 100, 2) if max_score > 0 else 0
        ),
        "status": report.status,
        "answered_count": answered_count,
        "unanswered_count": unanswered_count,
        "pending_count": pending_count,
    }


# NLP Suggestions Endpoint
@app.get(
    "/reports/{report_id}/topics/{topic_id}/suggestions/",
    response_model=List[Suggestion],
    tags=["NLP Suggestions"],
)
async def get_nlp_suggestions_endpoint(
    report_id: int,
    topic_id: int,
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    top_k_semantic: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    global sentence_model, nli_model, nli_tokenizer, NLI_ENTAILMENT_LABEL_ID

    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.status not in ["processed", "reviewing", "completed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Report NLP status is '{report.status}'. Suggestions available after 'processed'.",
        )

    topic_item = (
        db.query(ESGTopic).filter(ESGTopic.id == topic_id).first()
    )  # Renamed topic
    if not topic_item:
        raise HTTPException(status_code=404, detail="ESG Topic not found")

    if (
        sentence_model is None
        or nli_model is None
        or nli_tokenizer is None
        or NLI_ENTAILMENT_LABEL_ID == -1
    ):
        logger.warning(
            "NLP models not loaded (suggestions endpoint), attempting to load now."
        )
        load_nlp_models()
        if (
            sentence_model is None
            or nli_model is None
            or nli_tokenizer is None
            or NLI_ENTAILMENT_LABEL_ID == -1
        ):
            raise HTTPException(
                status_code=503,
                detail="NLP models are not available. Please try again later.",
            )

    hypothesis = topic_item.hypothesis_template
    logger.info(
        f"Generating suggestions for Report ID {report_id}, Topic ID {topic_id} ('{topic_item.name}') with threshold {threshold}, hypothesis: '{hypothesis}'"
    )

    report_chunks = (
        db.query(ReportChunk).filter(ReportChunk.report_id == report_id).all()
    )
    if not report_chunks:
        return []

    chunk_texts = [chunk.chunk_text for chunk in report_chunks]
    chunk_embeddings_list = [chunk.get_embedding() for chunk in report_chunks]
    chunk_coordinates_list = []

    for chunk in report_chunks:
        coordinates = None
        if chunk.coordinates_json:
            try:
                coordinates = json.loads(chunk.coordinates_json)
            except json.JSONDecodeError:
                pass
        chunk_coordinates_list.append(coordinates)

    valid_chunks_indices = [
        i for i, emb in enumerate(chunk_embeddings_list) if emb is not None
    ]
    if not valid_chunks_indices:
        logger.warning(f"No valid embeddings found for chunks in report {report_id}")
        return []

    filtered_chunk_texts = [chunk_texts[i] for i in valid_chunks_indices]
    filtered_report_chunks = [report_chunks[i] for i in valid_chunks_indices]
    filtered_chunk_coordinates = [
        chunk_coordinates_list[i] for i in valid_chunks_indices
    ]
    filtered_chunk_embeddings_np = np.array(
        [chunk_embeddings_list[i] for i in valid_chunks_indices], dtype=np.float32
    )

    if filtered_chunk_embeddings_np.ndim == 1:  # Handle case of single valid chunk
        filtered_chunk_embeddings_np = filtered_chunk_embeddings_np.reshape(1, -1)
    if filtered_chunk_embeddings_np.shape[0] == 0:  # No valid embeddings
        return []

    hypothesis_embedding = (
        sentence_model.encode(hypothesis, convert_to_tensor=True).cpu().numpy()
    )
    if hypothesis_embedding.ndim == 1:
        hypothesis_embedding = hypothesis_embedding.reshape(1, -1)

    from sentence_transformers.util import semantic_search

    hits = semantic_search(
        hypothesis_embedding,
        filtered_chunk_embeddings_np,
        top_k=min(top_k_semantic, filtered_chunk_embeddings_np.shape[0]),
    )[0]

    candidate_chunks_info = []
    for hit in hits:
        corpus_id = hit["corpus_id"]
        original_chunk = filtered_report_chunks[corpus_id]
        original_chunk_coords = filtered_chunk_coordinates[corpus_id]
        candidate_chunks_info.append(
            {
                "id": original_chunk.id,
                "text": original_chunk.chunk_text,
                "page_number": original_chunk.page_number,
                "semantic_score": hit["score"],
                "coordinates": original_chunk_coords,
            }
        )

    if not candidate_chunks_info:
        logger.info("No semantic candidates found.")
        return []

    logger.info(
        f"Found {len(candidate_chunks_info)} semantic candidates after semantic search."
    )

    suggestions = []
    nli_premises = [candidate["text"] for candidate in candidate_chunks_info]
    nli_hypotheses = [hypothesis] * len(nli_premises)

    if nli_premises:
        try:
            inputs = nli_tokenizer(
                nli_premises,
                nli_hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=nli_tokenizer.model_max_length,
            ).to(device)
            with torch.no_grad():
                outputs = nli_model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

            for idx, candidate_info in enumerate(candidate_chunks_info):
                entailment_prob = probabilities[idx][NLI_ENTAILMENT_LABEL_ID]
                if entailment_prob >= threshold:
                    suggestions.append(
                        Suggestion(
                            chunk_id=candidate_info["id"],
                            chunk_text=(
                                candidate_info["text"][:500] + "..."
                                if len(candidate_info["text"]) > 500
                                else candidate_info["text"]
                            ),
                            page_number=candidate_info["page_number"],
                            entailment_score=round(entailment_prob, 4),
                            coordinates=candidate_info["coordinates"],
                        )
                    )
        except Exception as e:
            logger.error(f"Error during NLI batch inference: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during NLI inference.")

    logger.info(
        "Generated %d suggestions passing NLI threshold %f.",
        len(suggestions),
        threshold
    )
    suggestions.sort(key=lambda s: s.entailment_score, reverse=True)
    return suggestions


# --- Main (for Uvicorn) ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ESG Scoring Assistant API...")
    # Optional: Clear DB on restart during dev. Use with caution.
    # Base.metadata.drop_all(bind=engine)
    # Base.metadata.create_all(bind=engine) # This is now in startup_event
    # with SessionLocal() as db_main: # Initialization is now in startup_event
    #     initialize_esg_topics(db_main)
    uvicorn.run(
        app, host="0.0.0.0", port=8000, reload=True
    )  # reload=True for development
