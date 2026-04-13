FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artefacts and service code
COPY anomaly_model.pkl scaler.pkl ml_service.py ./

EXPOSE 8000

CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
