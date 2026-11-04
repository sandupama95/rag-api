FROM python:3.11-slim
WORKDIR /code

RUN useradd -m -u 1000 appuser
ENV HF_HOME=/code/hf_cache

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
RUN mkdir -p /code/data && chown -R appuser:appuser /code
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]