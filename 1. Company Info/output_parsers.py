

from langchain.pydantic_v1 import BaseModel, Field



class CompanyInfoOutputParser(BaseModel):

    company_name: str = Field(description="The name of the company.")
    company_field: str = Field(description="The main field of the company.")
    company_country: str = Field(description="The country where the company was founded.")
    company_main_product: str = Field(description="The main product of the company.")
    




