from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector
from .database import Base
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector)  # Adjust embedding dimension if needed
