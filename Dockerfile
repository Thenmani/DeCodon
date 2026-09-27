FROM python:3.12-slim

WORKDIR /app

# Install dependencies directly for fast, reliable Cloud Run builds
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    jinja2 \
    python-dotenv \
    scikit-learn \
    numpy \
    google-genai

# Copy project files
COPY . .

# Cloud Run injects PORT (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]