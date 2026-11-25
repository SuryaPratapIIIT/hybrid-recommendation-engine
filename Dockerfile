FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 & python -m streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"]