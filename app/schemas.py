from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from typing import List 

class QuestionCreate(BaseModel):
    """Schema for creating a question."""
    title: str
    description: str
    asked_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ResumeDetailResponse(BaseModel):
    data: str
    error: Optional[str] = None


# class QnAResponse(BaseModel):
#     question: QuestionResponse
#     answers: List[AnswerResponse]  # List of answers to the question

#     class Config:
#         orm_mode = True