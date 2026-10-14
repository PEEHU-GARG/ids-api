FROM python:3.12-slim

WORKDIR /code

# Install dependencies first (better layer caching)
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy the app code and the trained model files
COPY app ./app
COPY models ./models

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}