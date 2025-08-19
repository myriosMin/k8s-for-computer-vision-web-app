# CPU-only version of infer service
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata python3-opencv git build-essential && \
    ln -fs /usr/share/zoneinfo/Asia/Singapore /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch and other dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

# Create required directories
RUN mkdir -p /output/predictions /models

ENV PYTHONPATH=/app

EXPOSE 9000

CMD ["uvicorn", "app.infer_api:app", "--host", "0.0.0.0", "--port", "9000"]
