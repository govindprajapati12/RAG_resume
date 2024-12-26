from http.client import HTTPException
from fastapi import UploadFile
from typing import Iterator
import os
from docling.document_converter import DocumentConverter
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, MarkdownFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from app.database import llm
from typing import Dict, Any
from langchain_postgres.vectorstores import PGVector
from app.database import embed,connection
from langchain_core.prompts import PromptTemplate

UPLOAD_DIR = "uploaded_files"

class DoclingFileLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend),
                InputFormat.MD: MarkdownFormatOption(pipeline_cls=SimplePipeline),
            },
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

def Read_upload_document_Embeding(file: UploadFile):
    """
    Upload a document, extract text using Docling, convert to embeddings, and store in the database.
    """

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())  # Write file content to disk


    try:
        loader = DoclingFileLoader(file_path)
        test_splitter = RecursiveCharacterTextSplitter(
            chunk_size= 1000,
            chunk_overlap=200
        )
        docs = loader.load()
        splits = test_splitter.split_documents(docs)
        return " ".join([split.page_content for split in splits])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")


def chunk_text(text: str, chunk_size: int = 1000):
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > chunk_size:
                if current_chunk:  # Avoid empty chunks
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks



class ConversationHandler:
    def __init__(self):
        """
        Initialize conversation handler with persistent memory
        
        Args:
            vectorstore: Initialized vector store for retrieval
        """
        collection_name = "my_docs"
        
        self.vectorstore = PGVector(
                embeddings=embed,
                collection_name=collection_name,
                connection=connection
            )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        
    def create_qa_chain(self):
        """
        Create a new conversational retrieval chain with custom prompt
        
        Returns:
            LangChain retrieval chain
        """
        
        # Configure advanced retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 3,  # Top 3 most relevant documents
                "fetch_k": 10,  # Search among top 10 initially
                "search_type": "mmr",  # Maximal Marginal Relevance for diverse results
            }
        )        
        # Create a custom retrieval QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True  # Detailed logging
        )
        
        return qa_chain
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced query processing with detailed source tracking
        
        Args:
            query (str): User's input question
        
        Returns:
            Dict with answer, sources, and context details
        """
        # Create a new chain for each query
        qa_chain = self.create_qa_chain()
        
        # Run the chain with the query 
        # Use 'question' key instead of 'input'
        result = qa_chain({"question": query})
        
        # Extract answer
        answer = result.get("answer", "No answer available.")
        
        # Process source documents with enhanced metadata
        sources = result.get("source_documents", [])
        processed_sources = [
            {
                "content": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                }
            } for doc in sources
        ]
        
        return {
            "answer": answer,
            "sources": processed_sources,
            "source_count": len(processed_sources),
            "chat_history": self.memory.chat_memory.messages
        }
    
    def clear_memory(self):
        """
        Clear the conversation memory
        """
        self.memory.clear()





class Resume:
    def __init__(self,userid):
        self.vectorstore = PGVector(
                embeddings=embed,
                collection_name=f"{userid}_resume",
                connection=connection
            )

    async def resume_context(self, file:UploadFile):
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())  # Write file content to disk
        try:
            loader = DoclingFileLoader(file_path)
            test_splitter = RecursiveCharacterTextSplitter(
                chunk_size= 1000,
                chunk_overlap=200
            )
            docs = loader.load()
            splits = test_splitter.split_documents(docs)
            return " ".join([split.page_content for split in splits])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")
    # This function handles the model invocation with a system prompt to extract data.
    def extract_data_with_prompt(self,userid,prompt: str) -> str:
        try:
            # Extract the text from the resume
            texts = self.vectorstore.similarity_search(f"{userid}_resume")
        except Exception as e:
            raise ValueError(f"Failed to process the file: {str(e)}")
        
        # Define the system prompt (the same prompt structure you need)
        formatted_prompt = prompt.format(resume_content=texts)
        
        # Assuming llm.invoke handles sending the prompt to the model and getting the result.
        try:
            response = llm.invoke(formatted_prompt)  # Ensure llm.invoke is set up properly
            return response
        except Exception as e:
            raise ValueError(f"Failed to generate response: {str(e)}")
        
    async def get_response_from_model(self,file : UploadFile ,userid):
        texts = await self.resume_context(file)  # Extract resume content as plain text
        chunks = chunk_text(text=texts)
        documents = [LCDocument(page_content=chunk) for chunk in chunks]
        # resume_obj = Resume(userid) 
        _ = self.vectorstore.add_documents(documents=documents)  # Split text logically (optional, if necessary)
        docs_content = "\n\n".join(chunks)  # Combine chunks for processing

        resume_summary_prompt = PromptTemplate.from_template(
            """
            You are a professional resume summarizer. Given the full content of a resume in a 6 lines, extract the most relevant information and present it in a concise and structured format for HR professionals. Include the following:

            - **Name:** [Full Name]
            - **Experience:** [Total Years of Work Experience]
            - **Key Skills:** [3-5 most relevant skills]
            - **Last Position Held:** [Job Title, Company Name]
            The output must be professional, easy to read, and provide a clear summary of the candidate's. Here is the full resume content:

            {resume_content}
            """
        )
        formatted_prompt = resume_summary_prompt.format(resume_content=docs_content)
        return formatted_prompt