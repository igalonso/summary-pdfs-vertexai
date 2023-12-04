from langchain.chains.summarize import load_summarize_chain
from langchain.llms import VertexAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import Docx2txtLoader as DocxLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import os
import json

def write_to_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)

def summarize_text(text):
    loader = TextLoader(text)
    llm = VertexAI(temperature=0.3, verbose=True, max_output_tokens=2047, model_name="text-bison-32k")
    text_loaded = loader.load()
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(text_loaded)
    return summary

def summarize_docx(filename):
    loader = DocxLoader(filename)
    document = loader.load_and_split()
    
    llm = VertexAI(temperature=0.3, verbose=True, max_output_tokens=2047, model_name="text-bison-32k")
    chain = load_summarize_chain(llm, chain_type="refine")
    summary = chain.run(document)
    return summary

def summarize_pdf(filename):
    loader = PyPDFLoader(filename)
    document = loader.load()
    llm = VertexAI(temperature=0.3, verbose=True, max_output_tokens=2047, model_name="text-bison-32k")
    chain = load_summarize_chain(llm, chain_type="refine")
    summary = chain.run(document,return_only_outputs=True)
    return summary


summary = summarize_docx("document-2.docx")
print(summary)
print("-"*17)
write_to_file("summary-2.txt", summary)
summary = summarize_text("summary-2.txt")
print(summary)

