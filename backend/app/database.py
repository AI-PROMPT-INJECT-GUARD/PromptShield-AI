"""
database.py
------------
SQLite storage for prediction history using SQLAlchemy.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./promptshield.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    prompt_text = Column(String, nullable=False)
    label = Column(String, nullable=False)
    is_injection = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    attack_category = Column(String, nullable=True)
    explanation = Column(String, nullable=True)
    safe_prompt = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()