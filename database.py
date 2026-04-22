from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://localhost/vehicle_detection"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"

    id          = Column(Integer, primary_key=True, index=True)
    filename    = Column(String)
    type        = Column(String)  # "image" or "video"
    total       = Column(Integer)
    detections  = Column(JSON)
    created_at  = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)