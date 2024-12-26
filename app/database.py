from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text  # Import `text` for raw SQL
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaLLM

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Set up database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Ensure pgvector extension is enabled
with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))  # Use `text`

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



embed = OllamaEmbeddings(
    model="llama3", 
    base_url="http://host.docker.internal:11434"
)

connection = os.getenv("DATABASE_URL")  # Use DATABASE_URL from the .env file
collection_name = "my_docs"
vectorstore = PGVector(
        embeddings=embed,
        collection_name=collection_name,
        connection=connection
    )
llm = OllamaLLM(
    base_url='http://host.docker.internal:11434', 
    model="llama3"
)