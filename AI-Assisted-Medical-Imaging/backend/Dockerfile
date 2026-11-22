FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

RUN mkdir -p /app/models /app/datasets /app/storage

COPY models/rsna_model.pth /app/models/rsna_model.pth
COPY data/rsna/stage_2_train_labels.csv /app/datasets/stage_2_train_labels.csv
COPY backend/ /app/backend/

ENV MODEL_PATH=/app/models/rsna_model.pth
ENV RSNA_LABELS_CSV=/app/datasets/stage_2_train_labels.csv
ENV STORAGE_DIR=/app/storage

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
