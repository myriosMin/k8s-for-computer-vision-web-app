# 1) Build Tailwind (no external network at runtime)
FROM node:22-bullseye AS web

WORKDIR /ui

COPY images/ui/app/templates/ ../templates/
COPY images/ui/app/static/js/ ../static/js/
COPY images/ui/app/static/css/tailwind.css ./tailwind.css
COPY images/ui/app/static/css/tailwind.config.js ./tailwind.config.js

# Tailwind CLI (no config = just default)
RUN npm init -y && npm i -D tailwindcss@^3 && \
    npx tailwindcss -i ./tailwind.css -o ./dist.css --minify


#####

# 2) Python runtime
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY images/ui/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY images/ui/app/ ./ 

# bring minified CSS
COPY --from=web /ui/dist.css ./static/css/dist.css
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "wsgi:app"]