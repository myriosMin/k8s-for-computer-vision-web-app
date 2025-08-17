# 1) Build Tailwind (no external network at runtime)
FROM node:22-bullseye AS web

WORKDIR /ui

COPY app/templates/ ../templates/
COPY app/static/js/ ../static/js/
COPY app/static/css/tailwind.css ./tailwind.css
COPY app/static/css/tailwind.config.js ./tailwind.config.js

# Tailwind CLI (no config = just default)
RUN npm init -y && npm i -D tailwindcss@^3 && \
    npx tailwindcss -i ./tailwind.css -o ./dist.css --minify

#####

# 2) Python runtime
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl && rm -rf /var/lib/apt/lists/*

# --- Install deps + kubectl + make ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    make \
 && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./

# bring minified CSS
COPY --from=web /ui/dist.css ./static/css/dist.css
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "wsgi:app"]