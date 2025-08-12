FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# Falls back to CPU if no GPU in Minikube; cuda 12.8, pytorch 2.7 for Jin Bin's 5070Ti GPU

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata python3-opencv git && \
    ln -fs /usr/share/zoneinfo/Asia/Singapore /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY images/worker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY images/worker/app/ ./app

ENV PYTHONPATH=/app

CMD ["bash","-lc","python -m app.preprocess && python -m app.train"]