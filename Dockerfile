FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies first to maximize layer caching.
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .

# Copy runtime assets and project files.
COPY data /app/data
COPY artifacts /app/artifacts
COPY tests /app/tests

# Default command is API for direct docker run usage.
EXPOSE 8000
CMD ["uvicorn", "creditrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
