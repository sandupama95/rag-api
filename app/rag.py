import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # small, free, fast
client = chromadb.PersistentClient(path="./data")
collection = client.get_or_create_collection("docs")

def chunk_text(text, size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap   # overlap keeps context across cuts
    return chunks

def add_document(doc_id, text):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    collection.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )

def retrieve(question, k=3):
    q_emb = model.encode([question]).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=k)
    return result["documents"][0]