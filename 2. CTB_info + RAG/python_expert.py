
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from retriever import Retriever

load_dotenv()

class PythonExpert:

    def __init__(self):
        self.__model = PythonExpert.__get_Anthropic()
        self.__prompt_template = PythonExpert.__get_prompt_template()

        pdf_list = ['Livro de Python (desenvolvedores).pdf']
        self.__retriever = Retriever(pdf_path_list=pdf_list)


    def get_answer(self, question:str):
        context_text = self.__retriever.retrieve(question)
        prompt = self.__prompt_template.format(context=context_text, text=question)
        response = self.__model.invoke(prompt)

        return response



    @staticmethod
    def __get_Anthropic():
        model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        return model
    

    @staticmethod
    def __get_prompt_template():
        template = """Voce é um especialista em python.
                Voce irá responder da maneira precisa e direto em suas respostas.
                Sempre tentar fornecer um exemplo de codigo em python que exemplifica a resposta.
                Voce  recebe uma pergunta sobre python.
                Fornece a resposta em um formato Json valido.
                A saida deve estar em um formato Json valido, com a chave 'resposta'.
                Se não souber a respostar a saída deve ser 'não sei responder'.

                Responda a pergunta com base apenas no seguinte contexto
                Contexto: {context}
                
                A pergunta   : {text}
                """
        prompt_template = ChatPromptTemplate.from_template(template)
        return prompt_template