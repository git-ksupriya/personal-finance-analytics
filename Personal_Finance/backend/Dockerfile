FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY ml/lstm /app/ml/lstm
COPY ml/lstm_categories/train_lstm.py /app/ml/lstm_categories/train_lstm.py
COPY ml/lstm_categories/preprocessing /app/ml/lstm_categories/preprocessing
COPY ml/lstm_categories/models /app/ml/lstm_categories/models
COPY ml/utils /app/ml/utils

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "backend.app:app"]
