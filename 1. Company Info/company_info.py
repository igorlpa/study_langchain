
import pprint
from langchain_core.prompts import ChatPromptTemplate
from chat_factory import get_Anthropic
from output_parsers import CompanyInfoOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 

load_dotenv()

class CompanyInfo:

    def __init__(self):
        self.model = get_Anthropic()
        self.template = """You are an international expert in company information.
                            You are precise and direct in your responses. 
                            You are given a company name.
                            Provide the full name of the company, field, country and main product in a valid Json Format output.
                            The output should be in a valid Json Format having the values in Portuguese and keys in english
                            
                            Output json keys:"company_name", "company_field","company_country","company_main_product"

                            Company name: {text}
                            """
        self.prompt_template = PromptTemplate(template=self.template, input_variables=["text"])
        self.parser = PydanticOutputParser(pydantic_object=CompanyInfoOutputParser)
        self.chain =  self.prompt_template | self.model | self.parser


    def get_company_info(self, company_name: str) -> dict:
        """Get company information.

        This function takes a company name as an argument and returns a dictionary containing the company's full name, field, country and main product.

        Args:
            company_name (str): The name of the company.

        Returns:
            dict: A dictionary containing the company's full name, field, country and main product.
        """
        response = self.chain.invoke({"text": company_name})

        # Return the response as a dictionary
        return response
    

