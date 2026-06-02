FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src

COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-optional.txt

COPY src ./src

# Runs the full comparison (downloads dataset + DistilBERT at runtime). Override CMD with
# --skip-bert for a faster RNN/LSTM-only run.
CMD ["python", "-m", "sentiment.compare", "--output", "reports"]
