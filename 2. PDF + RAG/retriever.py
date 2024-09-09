from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma



class Retriever:

    def __init__(self, pdf_path_list:list[str]):

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        self._embeddings = HuggingFaceEmbeddings()

        documents =[]
        for pdf_path in pdf_path_list:
            pdf_loader = PyPDFLoader(pdf_path)
            pages = pdf_loader.load()
            documents.extend(pages)

        texts = self._text_splitter.split_documents(documents)
        
        self._db = Chroma.from_documents(texts, self._embeddings)


    def retrieve(self, query:str):

        docs_chroma = self._db.similarity_search_with_score(query, k=8)
        context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

        return context_text