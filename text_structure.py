from langchain.text_splitter import RecursiveCharacterTextSplitter

text="""In publishing and graphic design, Lorem ipsum is a placeholder text commonly used to demonstrate the visual form of a document or a typeface without relying on meaningful content. Lorem ipsum may be used as a placeholder before the final copy is available."""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,     
    chunk_overlap=0,
)
chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)