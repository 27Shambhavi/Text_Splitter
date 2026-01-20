from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Use a stable sentence-transformer model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season.
The sun was bright, and the air smelled of earth and fresh grass.

The Indian Premier League (IPL) is the biggest cricket league in the world.
People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety.
It causes harm to people and creates fear in cities and villages.
"""

docs = text_splitter.create_documents([sample])

print(len(docs))
for i, d in enumerate(docs, 3):
    print(f"\n\n--- Chunk {i} ---")
    print(d.page_content)
