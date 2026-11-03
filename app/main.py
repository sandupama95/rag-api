from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from app import rag, llm
from fastapi import Depends
from app.llm import get_llm
from pypdf import PdfReader
from io import BytesIO
from fastapi import HTTPException

app = FastAPI(title="RAG API")

class Question(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}          # needed later by Docker and deployment

def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Upload .pdf, .txt or .md",
    )

@app.post("/documents")
async def upload(file: UploadFile):
    data = await file.read()
    text = extract_text(file.filename, data)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in file")
    rag.add_document(file.filename, text)
    return {"stored": file.filename, "characters": len(text)}

@app.post("/ask")
def ask(body: Question, llm = Depends(get_llm)):
    chunks = rag.retrieve(body.question)
    context = "\n---\n".join(chunks)
    prompt = (
        "Answer ONLY using the context below. "
        "If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {body.question}"
    )
    return {"answer": llm.generate(prompt), "sources": chunks}