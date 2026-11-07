# Reproducible, non-root runtime image (CPU). Build:  docker build -t rl-hedging .
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-py312.lock.txt requirements.lock.txt
COPY pyproject.toml ./
RUN pip install -r requirements.lock.txt
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps -e . && useradd -m app && chown -R app /app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address=0.0.0.0"]
