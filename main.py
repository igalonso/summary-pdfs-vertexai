from langchain.chains.summarize import load_summarize_chain
from langchain.llms import VertexAI
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv
load_dotenv()
#from google.api_core.client_options import ClientOptions
#from google.cloud import documentai  # type: ignore
import os
import json

loader = PyPDFLoader("document.pdf")
document = loader.load()
docs = loader.load()
llm = VertexAI(temperature=0.3, verbose=True, max_output_tokens=2047,model_name="text-bison-32k")
chain = load_summarize_chain(llm, chain_type="map_reduce")

summary = chain.run(docs)
print(summary)