from langchain.chains.summarize import load_summarize_chain
from langchain.llms import VertexAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader as DocxLoader

from dotenv import load_dotenv
load_dotenv()
#from google.api_core.client_options import ClientOptions
#from google.cloud import documentai  # type: ignore
import os
import json


def summarize_docx(filename):
    loader = DocxLoader(filename)
    document = loader.load()
    llm = VertexAI(temperature=0.3, verbose=True, max_output_tokens=2047, model_name="text-bison-32k")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(document)
    return summary


# loader = PyPDFLoader("document.pdf")
# document = loader.load()
# docs = loader.load()
# llm = VertexAI(temperature=0.3, verbose=True, max_output_tokens=2047,model_name="text-bison-32k")
# chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = summarize_docx("document.docx")
print(summary)