
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y     build-essential     libfreetype6-dev     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app_V2.py ./app_V2.py
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8501
CMD ["streamlit", "run", "app_V2.py", "--server.port=8501", "--server.address=0.0.0.0"]
