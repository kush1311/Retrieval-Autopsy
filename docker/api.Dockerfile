FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml ./
COPY autopsy ./autopsy
COPY corpus ./corpus
COPY evals ./evals
COPY api ./api

RUN pip install --no-cache-dir -e ".[llm]"

# Build the index at image build time. Ingest is deterministic and content-addressed,
# so baking it in makes container start instant and makes the corpus version part of
# the image identity rather than something that drifts at runtime.
RUN python -m autopsy.cli ingest

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
