FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 && rm -rf /var/lib/apt/lists/*
COPY images/infer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY images/infer/app/ ./app
ENV PYTHONPATH=/app
EXPOSE 9000
CMD ["python","-m","app.infer_api"]