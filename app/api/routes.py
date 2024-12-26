from fastapi import APIRouter, File, UploadFile,Form, Query
from .service import Read_upload_document_Embeding
from langchain_core.documents import Document
from app.database import vectorstore,llm
from .service import Resume,ConversationHandler,chunk_text
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
from app.api.prompt import *
from app.schemas import ResumeDetailResponse

router = APIRouter()

@router.post("/stores/")
def vectorestores_setup_with_docd(file: UploadFile = File(...)):
        # Read document and get embeddings
        texts = Read_upload_document_Embeding(file)
        chunks = chunk_text(text = texts)

        documents = [Document(page_content=chunk) for chunk in chunks]
        # Index chunks
        _ = vectorstore.add_documents(documents=documents)
        return {"message": "Document uploaded and embeddings stored successfully."}

@router.post("/ask/")
async def ask_question(
    query: str = Form(...)
):
    global conversation_handler
    conversation_handler = None
    # Initialize handler if not already created
    if conversation_handler is None:
        conversation_handler = ConversationHandler(vectorstore)
    # Process the query
    result = conversation_handler.process_query(query)
    return result

@router.post("/clear-history/")
async def clear_conversation_history():
    global conversation_handler
    
    if conversation_handler is not None:
        conversation_handler.clear_memory()
        return {"status": "Conversation history cleared"}
    
    return {"status": "No active conversation"}
  
@router.post('/resume')
async def resume_summary(userid:str = Query(...),file: UploadFile = File(...)):
    try:
            # Log incoming request details
            print(f"Received resume upload for user: {userid}")
            print(f"Filename: {file.filename}")
            print(f"Content type: {file.content_type}")

            # Your existing logic here
            resume_obj = Resume(userid)
            formatted_prompt = await resume_obj.get_response_from_model(file, userid)
            response = llm.invoke(formatted_prompt)
            print('.........................',response)
            return {"summary": response}
    except Exception as e:
        # Detailed error logging
        print(f"Error in resume upload: {str(e)}")
        # Return more detailed error response
        return {"error": str(e)}, 400 
     
# Endpoint for extracting overall experience from the resume
@router.get('/experience', response_model=ResumeDetailResponse)
def get_experience(userid:str = Query(...)):
    try:
        resume_obje = Resume(userid)
        experience_prompt = prompt_data["experience_prompt"]
        experience = resume_obje.extract_data_with_prompt(userid,experience_prompt)
        return ResumeDetailResponse(data=experience)
    except ValueError as e:
      return ResumeDetailResponse(error=str(e))

# Endpoint for extracting the working company from the resume
@router.get('/working_company', response_model=ResumeDetailResponse)
def get_working_company(userid:str =Query(...)):
    try:
        resume_obje = Resume(userid)
        company_prompt = prompt_data["company_prompt"]
        company = resume_obje.extract_data_with_prompt(userid, company_prompt)
        return ResumeDetailResponse(data=company)
    except ValueError as e:
        return ResumeDetailResponse(error=str(e))

# Endpoint for extracting key skills from the resume
@router.get('/skills', response_model=ResumeDetailResponse)
def get_skills(userid:str = Query(...)):
    try:
        resume_obje = Resume(userid)
        skills_prompt = prompt_data["skills_prompt"]
        skills = resume_obje.extract_data_with_prompt(userid, skills_prompt)
        return ResumeDetailResponse(data=skills)
    except ValueError as e:
        return ResumeDetailResponse(error=str(e))

# Endpoint for extracting education information from the resume
@router.get('/education', response_model=ResumeDetailResponse)
def get_education(userid: str = Query(...)):
    try:
        resume_obje = Resume(userid)
        education_prompt = prompt_data["education_prompt"]
        education = resume_obje.extract_data_with_prompt(userid, education_prompt)
        return ResumeDetailResponse(data=education)
    except ValueError as e:
        return ResumeDetailResponse(error=str(e))

# Endpoint for extracting notable achievements from the resume
@router.get('/achievements', response_model=ResumeDetailResponse)
def get_achievements(userid: str = Query(...)):
    try:
        resume_obje = Resume(userid)
        achievements_prompt = prompt_data["achievements_prompt"] 
        achievements = resume_obje.extract_data_with_prompt(userid, achievements_prompt)
        return ResumeDetailResponse(data=achievements)
    except ValueError as e:
        return ResumeDetailResponse(error=str(e))
