from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Building Machine Learning Systems with Python - Second Edition.pdf")
docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
) 

result=splitter.split_documents(docs)
print(result[0])