
from langchain.pydantic_v1 import BaseModel, Field



class AnswerOutPut(BaseModel):

    answer: str = Field(description="Resposta da pergunta.")
    confidence: float = Field(description="Confiança da resposta.")

