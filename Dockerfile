FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/ShreesthaHossain/customer-churn-intelligence

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    LOG_JSON=true \
    SERVICE_MODE=api

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

COPY . .

RUN python scripts/bootstrap_artifacts.py && chmod +x scripts/docker_entrypoint.sh

EXPOSE 8000

CMD ["sh", "scripts/docker_entrypoint.sh"]
